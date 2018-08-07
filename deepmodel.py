#encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import argparse
import pickle
import tensorflow as tf
from embedding import Embedding
from raw2std import Vocabulary
import numpy as np
import os.path as osp
tf.logging.set_verbosity(tf.logging.INFO)

class BiLSTMCRF(object):
    
    def __init__(self,config,test_mode=False):
        self.config = config
        self.test_mode = test_mode
        self.initializer = tf.contrib.layers.xavier_initializer(
            uniform=True, seed=None, dtype=tf.float32)

    def build(self,emb_weights=None):
        self.build_batch_input()
        self.build_embeddings(emb_weights)
        self.build_lstm_model()
        self.build_sentence_score_loss()
        if not self.test_mode:
            self.build_optimizer()

    def build_batch_input(self):
        if not self.test_mode:
            data_file = 'output/data/PKU1998_01_train.tfrecords'
            def _parse_wrapper(l):
                    return parse_example_queue(l)
            dataset = tf.data.TFRecordDataset(data_file).map(
                _parse_wrapper)
            with tf.name_scope('inputs'):
                dataset = dataset.shuffle(
                        buffer_size=256).repeat().shuffle(buffer_size=256)
                dataset = dataset.padded_batch(batch_size=self.config['batch_size'],
                                            padded_shapes=(tf.TensorShape([self.config['padded_len']]),
                                                            tf.TensorShape(
                                                                [self.config['padded_len']]),
                                                            tf.TensorShape([]))).filter(
                    lambda x, y, z: tf.equal(tf.shape(x)[0], self.config['batch_size']))
                
                iterator = dataset.make_one_shot_iterator()
                self.batch_input, self.tag_seqs, self.sequence_length = iterator.get_next()
                print(self.batch_input.shape, self.sequence_length.shape)
        else:
            with tf.name_scope('inputs_test'):
                    self.unpadded_input = tf.placeholder(
                        tf.int32, shape=[None])
                    self.seq_length = tf.placeholder(
                        tf.int32)
                    padding = tf.constant([0]*3)
                    padding = tf.concat(
                        (padding, [self.config['padded_len']-self.seq_length]), 0)
                    padding = tf.reshape(padding,[2,2])
                    self.batch_input = tf.expand_dims(self.unpadded_input, 0)
                    self.batch_input = tf.pad(self.batch_input,padding)
                    self.batch_input = self.batch_input[0]
                    self.batch_input = tf.expand_dims(self.batch_input, 0)
                    self.sequence_length = tf.expand_dims(
                        self.seq_length, 0)
                    

    def build_embeddings(self, emb_weights):
        # load pretrained embedding
        with tf.device('/gpu:0'):
            with tf.name_scope('embeddings'):
                self.embeddings = tf.constant(emb_weights)
                print(self.embeddings.shape)
                self.embed = tf.nn.embedding_lookup(
                    self.embeddings, self.batch_input)
                self.embed = tf.cast(self.embed,tf.float32)


    def build_lstm_model(self):
        """
        Build model.
        Returns:
            self.logit: A tensor containing the probability of prediction with the shape of [batch_size, padding_size, num_tag]
        """
        with tf.device('/gpu:0'):
            #Setup LSTM Cell
            fw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                num_units=self.config['num_lstm_units'], state_is_tuple=True)
            bw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                num_units=self.config['num_lstm_units'], state_is_tuple=True)

            #Dropout when training
            if not self.test_mode:
                fw_lstm_cell = tf.contrib.rnn.DropoutWrapper(
                    fw_lstm_cell,
                    input_keep_prob=self.config['lstm_dropout_keep_prob'],
                    output_keep_prob=self.config['lstm_dropout_keep_prob'])
                bw_lstm_cell = tf.contrib.rnn.DropoutWrapper(
                    bw_lstm_cell,
                    input_keep_prob=self.config['lstm_dropout_keep_prob'],
                    output_keep_prob=self.config['lstm_dropout_keep_prob'])

        
        self.embed.set_shape([None, None, self.config['embedding_size']])
        
        with tf.variable_scope('seq_lstm') as lstm_scope:
            #Run LSTM with sequence_length timesteps
            bi_output, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_lstm_cell,
                cell_bw=bw_lstm_cell,
                inputs=self.embed,
                sequence_length=self.sequence_length,
                dtype=tf.float32,
                scope=lstm_scope)
            fw_out, bw_out = bi_output
            lstm_output = tf.concat([fw_out, bw_out], 2)
        self.lstm_output = lstm_output
        return self.lstm_output

    def build_sentence_score_loss(self):
        """
        Use CRF log likelihood to get sentence score and loss
        """
        #Fully connected layer to get logit
        with tf.variable_scope('logit'):
            logit = tf.contrib.layers.fully_connected(
                inputs=self.lstm_output,
                num_outputs=self.config['num_tag'],
                activation_fn=None,
                weights_initializer=self.initializer)
        self.logit = logit

        if self.test_mode:
            with tf.variable_scope('tag_inf') as tag_scope:
                transition_param = tf.get_variable(
                    'transitions',
                    shape=[self.config['num_tag'], self.config['num_tag']])
        else:
            with tf.variable_scope('tag_inf') as tag_scope:
                sentence_likelihood, transition_param = tf.contrib.crf.crf_log_likelihood(
                    inputs=logit,
                    tag_indices=tf.to_int32(self.tag_seqs),
                    sequence_lengths=self.sequence_length)

        self.predict_tag, _ = tf.contrib.crf.crf_decode(
            logit, transition_param, self.sequence_length)
        
        if not self.test_mode:
            with tf.variable_scope('loss'):
                batch_loss = tf.reduce_mean(-sentence_likelihood)

                #Add to total loss
                tf.losses.add_loss(batch_loss)

                #Get total loss
                total_loss = tf.losses.get_total_loss()

                tf.summary.scalar('batch_loss', batch_loss)
                tf.summary.scalar('total_loss', total_loss)

        if not self.test_mode:
            with tf.variable_scope('accuracy'):

                seq_len = tf.cast(
                    tf.reduce_sum(self.sequence_length), tf.float32)
                padded_len = tf.cast(
                    tf.reduce_sum(
                        self.config['batch_size'] * self.config['padded_len']),
                    tf.float32)

                # Calculate acc
                correct = tf.cast(
                    tf.equal(self.predict_tag, tf.cast(self.tag_seqs,
                                                        tf.int32)), tf.float32)
                correct = tf.reduce_sum(correct) - padded_len + seq_len

                self.accuracy = correct / seq_len

        if not self.test_mode:
            
            tf.summary.scalar('average_len',
                                tf.reduce_mean(self.sequence_length))
            tf.summary.scalar('train_accuracy', self.accuracy)

            #Output loss
            self.batch_loss = batch_loss
            self.total_loss = total_loss
        
        self.merged = tf.summary.merge_all()
    # def setup_global_step(self):
    #     """Sets up the global step Tensor."""
    #     global_step = tf.Variable(
    #         initial_value=0,
    #         name="global_step",
    #         trainable=False,
    #         collections=[
    #             tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES
    #         ])
    #     self.global_step = global_step

    def build_optimizer(self):
        with tf.name_scope('optimizer'):
                # self.optimizer = tf.train.GradientDescentOptimizer(
                #     0.01).minimize(self.total_loss)
                self.optimizer = tf.train.AdamOptimizer(
                    0.001).minimize(self.total_loss)

def embedding_from_POLYGLOT(config):
    dictionary = pickle.load(open('output/dictionary.pkl','rb'))
    embedding = pickle.load(open('./polyglot-zh_char.pkl','rb'))
    vocab = Vocabulary(dictionary)
    em_weights = load_embedding(vocab,embedding,config['embedding_size'])
    return em_weights

def embbeding_from_scratch():
    path = './output/char_embedding.npy'
    assert osp.exists(path),'please train embedding first.'
    embedding = np.load(path)
    return embedding


def main():
    current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(current_path, 'output/model'),
        help='The log directory for TensorBoard summaries.')
    parser.add_argument('-r', '--resume', action='store_true',
                            help='resume from latest training')
    parser.add_argument('-e','--embedding_from_scratch',action='store_true',help='train embbeding from scratch.(default: False)')
    FLAGS, unparsed = parser.parse_known_args()
    # Create the directory for TensorBoard variables if there is not.
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    vocabulary_size = 3500
    embedding_size = 64  # Dimension of the embedding vector.
    skip_window = 1  # How many words to consider left and right.
    num_skips = 2  # How many times to reuse an input to generate a label.
    num_sampled = 64  # Number of negative examples to sample.
    valid_size = 16  # Random set of words to evaluate similarity on.
    # Only pick dev samples in the head of the distribution.
    valid_window = 100
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    # embedding_model = Embedding(
    #     vocabulary_size, batch_size, embedding_size, num_sampled, valid_examples)
    # embedding_model.build_graph()
    # graph,init,saver = embedding_model.graph,embedding_model.init,embedding_model.saver
    # embeddings = embedding_model.embeddings
    # with tf.Session(graph=graph) as session:
    #     init.run()
    #     print('Initialized')
    #     saver.restore(session, os.path.join('log', 'model.ckpt'))
    #     em_weights = embeddings.eval()
    #     print('resumed from latest checkpoint')

    config = {'batch_size': 512, 'embedding_size': 64, 'num_lstm_units': 128,
              'lstm_dropout_keep_prob': 0.35, 'padded_len': 60,'num_tag':5}
    
    if not FLAGS.embedding_from_scratch:
        em_weights = embedding_from_POLYGLOT(config)
    else:
        em_weights = embbeding_from_scratch()
    
    # em_weights = np.random.uniform(-1.0,1.0,(vocabulary_size,config['embedding_size']))

    print(em_weights.shape)
    graph = tf.Graph()
    with graph.as_default():
        estimate_model = BiLSTMCRF(config)
        estimate_model.build(em_weights)
        saver = tf.train.Saver()
        #Set up learning rate and learning rate decay function
        average_loss = 0
        with tf.Session(graph=graph) as session:
            num_steps = 100000
            tf.global_variables_initializer().run()
            writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)
            if FLAGS.resume:
                saver.restore(session, os.path.join(
                    FLAGS.log_dir, 'estimator.ckpt'))
                print('resumed from latest checkpoint')
            for step in xrange(1, num_steps):
                _,loss_val,acc_val,summary = session.run(
                    [estimate_model.optimizer, estimate_model.total_loss,estimate_model.accuracy,estimate_model.merged])
                print(step, loss_val, acc_val)
                writer.add_summary(summary, step)
                average_loss += loss_val
                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0
                    saver.save(session, os.path.join(
                        FLAGS.log_dir, 'estimator.ckpt'))
            writer.close()

def parse_example_queue(example_queue):
    """ Read one example.
      This function read one example and return context sequence and tag sequence
      correspondingly. 
      Args:
        filename_queue: A filename queue returned by string_input_producer
        context_feature_name: Context feature name in TFRecord. Set in ModelConfig
        tag_feature_name: Tag feature name in TFRecord. Set in ModelConfig
      Returns:
        input_seq: An int32 Tensor with different length.
        tag_seq: An int32 Tensor with different length.
      """

    #Parse one example
    context, features = tf.parse_single_sequence_example(
        example_queue,
        context_features={
            'length': tf.FixedLenFeature([], dtype=tf.int64)
        },
        sequence_features={
            'content_id':
            tf.FixedLenSequenceFeature([], dtype=tf.int64),
            'tag_id':
            tf.FixedLenSequenceFeature([], dtype=tf.int64)
        })

    return (features['content_id'],
            features['tag_id'], context['length'])

def load_embedding(vocab,original_embedding,embedding_size):
    #Init 2d numpy array
    embedding_table = np.zeros((len(vocab._vocab), embedding_size))

    word, embedding = original_embedding

    found = 0
    for i, w in enumerate(word):
        if w in vocab._vocab:
            found += 1
        embedding_table[vocab.word_to_id(w), :] = embedding[i, :]
    print('original vocabulary length: ',len(embedding))
    print('our vocabulary length: ',len(vocab._vocab))
    print('found:',found)
    print('word[0]:',word[0])
    #Manually set the last row of embedding(unknown chr)
    embedding_table[vocab._unk_id, :] = embedding[0, :]
    return embedding_table

if __name__ == '__main__':
    main()
