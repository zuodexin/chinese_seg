#encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import re
import pickle
import tensorflow as tf
import collections




class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self, vocab, unk_word='UNK'):
        """Initializes the vocabulary.
    Args:
      vocab: A dictionary of word to word_id.
      unk_id: Id of the special 'unknown' word.
    """
        self._vocab = vocab
        self._id_vocab = dict(zip(vocab.values(), vocab.keys()))
        self._unk_id = vocab[unk_word]

    def word_to_id(self, word):
        """Returns the integer id of a word string."""
        if word in self._vocab:
            return self._vocab[word]
        else:
            return self._unk_id

    def id_to_word(self, word_id):
        """Returns the word string of an integer word id."""
        if word_id >= len(self._vocab):
            return self._id_vocab[self._unk_id]
        else:
            return self._id_vocab[word_id]


def get_tag(word):
    if len(word)==1:
        return ['S']
    elif len(word)>=2:
        rst = ['B']
        for i in xrange(len(word)-2):
            rst.append('M')
        rst.append('E')
        return rst


def append_word(fp,word,t):
    tag_l = get_tag(word)
    chr_l = [ word[i].encode('utf-8') for i in xrange(len(word))]
    for i in xrange(len(word)):
        line = '{}\t{}\t{}\n'.format(chr_l[i],t,tag_l[i])
        fp.write(line)
    

def PKU1998_01_to_CRFPP():
    raw_fp = open('datasets/PKU1998_01/data.txt', 'r')
    train_fp = open('output/data/train.txt','w')
    test_fp = open('output/data/test.txt','w')
    print('transform raw data to CRF++ format...')
    lines = raw_fp.readlines()
    for i,line in enumerate(lines):
        progress = i/len(lines)
        update_progress(progress)
        line = line.decode('gbk')
        words = line.strip('\r\n\t').split()
        # print line
        if i%10==0:
            phase_fp = test_fp
        else:
            phase_fp = train_fp
        for word in words[1:]:
            # remove entity
            i1 = word.find('[')
            if i1>=0 and word[i1+1]!='/':
                word = word[i1+1:]
            i2 = word.find(']')
            if i2>=0 and i2+1<len(word) and word[i2+1]!='/':
                word = word[:i2]
            w,t = word.split('/')
            pingyin = re.compile(r'\{.*?\}')
            w = pingyin.sub('',w)
            # print(w)
            append_word(phase_fp,w,t)
    train_fp.close()
    test_fp.close()


def write_sequence_example(writer,decoded_str, pos_tag_str, vocab,max_len):
    #Transfer word to word_id
    content_id = [vocab.word_to_id(c) for c in decoded_str]
    tag_id = [tag_to_id(t) for t in pos_tag_str]
    # print('num chunks',len(range(0, len(content_id), max_len)))
    for i in xrange(0, len(content_id),max_len):
        content_id_chunk = content_id[i:i+max_len]
        tag_id_chunk = tag_id[i:i+max_len]
        length = len(content_id_chunk)
        feature_lists = tf.train.FeatureLists(feature_list={
            "content_id":
            _int64_feature_list(content_id_chunk),
            "tag_id":
            _int64_feature_list(tag_id_chunk)
        })

        context = tf.train.Features(feature={"length": _int64_feature(length)})

        sequence_example = tf.train.SequenceExample(
            feature_lists=feature_lists, context=context)
        writer.write(sequence_example.SerializeToString())

# Read the data into a string.
def read_data(filename):
    data = []
    with open(filename) as f:
        for line in f.readlines():
            data.extend(u'，'.join(line.decode('utf-8').strip().split()))
    print('total characters:',len(data))
    return data


def build_dict(words,n_words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reversed_dictionary



def PKU1998_01_to_tf_record():
    raw_fp = open('datasets/PKU1998_01/data.txt', 'r')
    embedding_train_data = read_data('output/merged.txt')
    vocabulary_size = 3500
    dictionary,reversed_dictionary = build_dict(embedding_train_data,vocabulary_size)
    pickle.dump(dictionary, open('output/dictionary.pkl', 'wb'))
    pickle.dump(reversed_dictionary, open(
        'output/reversed_dictionary.pkl', 'wb'))
    vocab = Vocabulary(dictionary)
    train_writer = tf.python_io.TFRecordWriter('output/data/PKU1998_01_train.tfrecords')
    test_writer = tf.python_io.TFRecordWriter('output/data/PKU1998_01_test.tfrecords')
    print('transform raw data to tf_recods...')
    tags_lookup = []
    for i in range(0, 300):
        if i <= 1:
            tags_lookup.append('S')
        else:
            tags_lookup.append('B' + 'M' * (i - 2) + 'E')
    lines = raw_fp.readlines()
    for i, line in enumerate(lines):
        progress = i/len(lines)
        update_progress(progress)
        line = line.decode('gbk')
        words = line.strip('\r\n\t').split()
        # print line
        if i % 10 == 0:
            tf_writer = test_writer
        else:
            tf_writer = train_writer
        cleaned_line = []
        tags = []
        for word in words[1:]:
            # remove entity
            i1 = word.find('[')
            if i1 >= 0 and word[i1+1] != '/':
                word = word[i1+1:]
            i2 = word.find(']')
            if i2 >= 0 and i2+1 < len(word) and word[i2+1] != '/':
                word = word[:i2]
            w, t = word.split('/')
            pingyin = re.compile(r'\{.*?\}')
            w = pingyin.sub('', w)
            cleaned_line.append(w)
            tags.append(tags_lookup[len(w)])
            # print(len(w),tags_lookup[len(w)])
        cleaned_line = ''.join(cleaned_line)
        tags = ''.join(tags)
        # print(cleaned_line,'\n',tags)
        if len(cleaned_line) != len(tags):
            print('Skip one row.' + ';' + cleaned_line)
            continue
        if len(cleaned_line) > 0:
            write_sequence_example(tf_writer,cleaned_line, tags, vocab,60)

    test_writer.close()
    train_writer.close()
    raw_fp.close()

def main():
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    PKU1998_01_to_CRFPP()
    PKU1998_01_to_tf_record()


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def tag_to_id(t):
    if t == 'S':
        return 1
    elif t == 'B':
        return 2
    elif t == 'E':
        return 3
    elif t == 'M':
        return 4

def id_to_tag(cid):
    d ='SSBEM'
    return d[cid]

def update_progress(progress):
    num = int(progress*10)
    print('\r[{0}{1}] {2:.2f}%'.format('#'*num,'-'*(10-num),progress*100))


if __name__ == '__main__':
    main()
