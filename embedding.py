#encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import collections
import argparse
import numpy as np
import random
import math
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import pickle

def print_ch(ch):
    for c in ch:
        print(c,end='')
    print('\n')

# Read the data into a string.
def read_data(filename):
    data = []
    with open(filename) as f:
        for line in f.readlines():
            data.extend(u'，'.join(line.decode('utf-8').strip().split()))
    print('total characters:',len(data))
    return data


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def char_seq_to_id_seq(char_seq,dictionary):
    id_seq = list()
    for c in char_seq:
        index = dictionary.get(c, 0)
        id_seq.append(index)
    return id_seq



# Step 3: Function to generate a training batch for the skip-gram model.
class BatchGenerator(object):
    def __init__(self,data,batch_size=8, num_skips=2, skip_window=1):
        self.order_index = 0
        self.data = data
        self.batch_size = batch_size
        self.num_skips = num_skips
        self.skip_window = skip_window
        self.order = range(len(data)-2 * self.skip_window - 1)
        random.shuffle(self.order)
    
    def generate_batch(self):
        data = self.data
        batch_size = self.batch_size
        num_skips = self.num_skips
        skip_window = self.skip_window
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        # buffer = collections.deque(maxlen=span)
        buffer = []
        if self.order_index >= len(self.order):
            self.order_index = 0
            random.shuffle(self.order)
        data_index = self.order[self.order_index]
        buffer = data[data_index:data_index + span]
        for i in range(batch_size // num_skips):
            context_words = [w for w in range(span) if w != skip_window]
            words_to_use = random.sample(context_words, num_skips)
            for j, context_word in enumerate(words_to_use):
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[context_word]
        self.order_index += 1
        return batch, labels

class Embedding(object):
    
    def __init__(self, vocabulary_size, batch_size, embedding_size, num_sampled, valid_examples):
        self.vocabulary_size = vocabulary_size
        self.batch_size = batch_size
        # Dimension of the embedding vector.
        self.embedding_size = embedding_size
        # Number of negative examples to sample.
        self.num_sampled = num_sampled
        self.valid_examples = valid_examples
        pass

    def build_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            # Input data.
            with tf.name_scope('inputs'):
                train_inputs = tf.placeholder(
                    tf.int32, shape=[self.batch_size])
                train_labels = tf.placeholder(
                    tf.int32, shape=[self.batch_size, 1])
                valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

            # Ops and variables pinned to the CPU because of missing GPU implementation
            with tf.device('/gpu:0'):
                # Look up embeddings for inputs.
                with tf.name_scope('embeddings'):
                    embeddings = tf.Variable(
                        tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
                    embed = tf.nn.embedding_lookup(
                        embeddings, train_inputs)

                # Construct the variables for the NCE loss
                with tf.name_scope('weights'):
                    nce_weights = tf.Variable(
                        tf.truncated_normal(
                            [self.vocabulary_size, self.embedding_size],
                            stddev=1.0 / math.sqrt(self.embedding_size)))
                with tf.name_scope('biases'):
                    nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            # Explanation of the meaning of NCE loss:
            #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
            with tf.name_scope('loss'):
                loss = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights=nce_weights,
                        biases=nce_biases,
                        labels=train_labels,
                        inputs=embed,
                        num_sampled=self.num_sampled,
                        num_classes=self.vocabulary_size))

            # Add the loss value as a scalar to summary.
            tf.summary.scalar('loss', loss)

            # Construct the SGD optimizer using a learning rate of.
            with tf.name_scope('optimizer'):
                # learning_rate = tf.train.exponential_decay(
                #     0.01,                # Base learning rate.
                #     batch * BATCH_SIZE,  # Current index into the dataset.
                #     train_size,          # Decay step.
                #     0.95,                # Decay rate.
                #     staircase=True)
                # optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
                optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                                        valid_dataset)
            similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True)

            self.loss = loss
            self.optimizer = optimizer
            # Merge all summaries.
            self.merged = tf.summary.merge_all()

            # Add variable initializer.
            self.init = tf.global_variables_initializer()

            # Create a saver.
            self.saver = tf.train.Saver()
        self.graph = graph
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.similarity = similarity
        self.normalized_embeddings = normalized_embeddings
        self.embeddings = embeddings


def train(model,batch_generator,num_steps,valid_size, valid_window, valid_examples,reverse_dictionary,log_dir,resume):
    model.build_graph()
    graph,init,merged = model.graph,model.init,model.merged
    loss, optimizer,saver = model.loss,model.optimizer,model.saver
    train_inputs, train_labels, similarity = model.train_inputs, model.train_labels, model.similarity
    normalized_embeddings, embeddings = model.normalized_embeddings,model.embeddings
    with tf.Session(graph=graph) as session:
        # Open a writer to write summaries.
        writer = tf.summary.FileWriter(log_dir, session.graph)
        # We must initialize all variables before we use them.
        init.run()
        print('Initialized')
        if resume:
            saver.restore(session, os.path.join(log_dir, 'model.ckpt'))
            print('resumed from latest checkpoint')
        average_loss = 0
        for step in xrange(0,num_steps):
            batch_inputs, batch_labels = batch_generator.generate_batch()
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # Define metadata variable.
            run_metadata = tf.RunMetadata()

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
            # Feed metadata variable to session for visualizing the graph in TensorBoard.
            _, summary, loss_val = session.run(
                [optimizer, merged, loss],
                feed_dict=feed_dict,
                run_metadata=run_metadata)
            average_loss += loss_val

            # Add returned summaries to writer in each step.
            writer.add_summary(summary, step)
            # Add metadata to visualize the graph for the last run.
            if step == (num_steps - 1):
                writer.add_run_metadata(run_metadata, 'step%d' % step)

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in xrange(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)

            if step % (num_steps//10) == 0:
                final_embeddings = normalized_embeddings.eval()
                print('saving snapshot at epoch ',step)
                # Write corresponding labels for the embeddings.
                with open(log_dir + '/metadata.tsv', 'w') as f:
                    for i in xrange(model.vocabulary_size):
                        f.write(reverse_dictionary[i].encode('utf-8') + '\n')
                # Save the model for checkpoints.
                saver.save(session, os.path.join(log_dir, 'model.ckpt'))
                # Create a configuration for visualizing embeddings with the labels in TensorBoard.
                config = projector.ProjectorConfig()
                embedding_conf = config.embeddings.add()
                embedding_conf.tensor_name = embeddings.name
                embedding_conf.metadata_path = os.path.join(log_dir, 'metadata.tsv')
                projector.visualize_embeddings(writer, config)
                np.save('./output/char_embedding.npy',final_embeddings)
                try:
                    from sklearn.manifold import TSNE
                    tsne = TSNE(
                        perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
                    plot_only = 500
                    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
                    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
                    plot_with_labels(low_dim_embs, labels, os.path.join('.', 'tsne.png'))
                except ImportError as ex:
                    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
                    print(ex)
        final_embeddings = normalized_embeddings.eval()
        writer.close()
    return final_embeddings

# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')
    plt.savefig(filename)


def main():
    # Give a folder path as an argument with '--log_dir' to save
    # TensorBoard summaries. Default is a log folder in current directory.
    current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(current_path, 'output/embedding_log'),
        help='The log directory for TensorBoard summaries.')
    parser.add_argument('-d','--data',type=str,help='The dataset file which contains Chinese strings',required=True)
    parser.add_argument('-r','--resume',action='store_true',help='resume from latest training')
    FLAGS, unparsed = parser.parse_known_args()

    # Create the directory for TensorBoard variables if there is not.
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    char_seq = read_data(FLAGS.data)

    # Step 2: Build the dictionary and replace rare words with UNK token.
    vocabulary_size = 3500
    # data, count, dictionary, reversde_dictionary = build_dataset(
    #     vocabulary, vocabulary_size)
    # pickle.dump(dictionary, open('output/dictionary.pkl', 'wb'))
    # pickle.dump(reversed_dictionary, open(
    #     'output/reversed_dictionary.pkl', 'wb'))
    try:
        dictionary =  pickle.load(open('output/dictionary.pkl','rb'))
        reversed_dictionary =  pickle.load(open('output/reversed_dictionary.pkl','rb'))
        data = char_seq_to_id_seq(char_seq,dictionary)
    except IOError as e:
        print(e)
    
    # print('Most common words (-UNK)',end='')
    # print(''.join([c[0] for c in count[0:10]]))
    # print('Sample data', data[:10], [reversed_dictionary[i] for i in data[:10]])

    batch_generator = BatchGenerator(
        data, batch_size=8, num_skips=2, skip_window=1)
    batch, labels = batch_generator.generate_batch()
    batch, labels = batch_generator.generate_batch()
    for i in range(8):
        print(batch[i], reversed_dictionary[batch[i]], '->', labels[i, 0],
        reversed_dictionary[labels[i, 0]])
    
    # Step 4: Build and train a skip-gram model.
    batch_size = 512
    embedding_size = 64  # Dimension of the embedding vector.
    skip_window = 1  # How many words to consider left and right.
    num_skips = 2  # How many times to reuse an input to generate a label.
    num_sampled = 64  # Number of negative examples to sample.
    
    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent. These 3 variables are used only for
    # displaying model accuracy, they don't affect calculation.
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    num_steps = 1000001
    batch_generator = BatchGenerator(
        data, batch_size=batch_size, num_skips=num_skips, skip_window=skip_window)
    embedding_model = Embedding(vocabulary_size,batch_size,embedding_size,num_sampled,valid_examples)

    final_embeddings = train(embedding_model, batch_generator, num_steps, valid_size,
          valid_window, valid_examples, reversed_dictionary, FLAGS.log_dir,FLAGS.resume)
        
if __name__ == '__main__':
    main()
