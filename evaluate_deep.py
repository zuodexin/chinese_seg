#encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import pickle
import tensorflow as tf
import numpy as np
from deepmodel import BiLSTMCRF
from raw2std import Vocabulary,id_to_tag
from deepmodel import load_embedding,parse_example_queue,embbeding_from_scratch,embedding_from_POLYGLOT

def eval_performance(gt,pred):
    wc_gt = 0
    wc_pred = 0
    wc_correct = 0
    flag = True
    for i in range(len(gt)):
        # print gt[i],tagger.y2(i)
        g,p= gt[i],pred[i]
        if g!=p:
            flag = False
        if p in ['E','S']:
            wc_pred += 1
            if flag:
                wc_correct += 1
            flag = True
        
        if g in ['E','S']:
            wc_gt += 1
    P = wc_correct*1.0/wc_pred
    R = wc_correct*1.0/wc_gt
    print('num words of predicted: ',wc_pred)
    print('num words of ground truth: ',wc_gt)
    print('num words of correct: ',wc_correct)
    print("P = %f, R = %f, F-score = %f" % (P, R, (2*P*R)/(P+R)))

def seq2batches(seq_str,vocab,max_len):
    content_id = [vocab.word_to_id(c) for c in seq_str]
    # print('num chunks', len(range(0, len(content_id), max_len)))
    for i in xrange(0, len(content_id), max_len):
        content_id_chunk = content_id[i:i+max_len]
        length = len(content_id_chunk)
        yield (content_id_chunk,length)



def test(modelpath,datapath,vocab,polyglot=True):
    output = open('./output/deep_partitioned.txt','w')
    config = {'batch_size': 1, 'embedding_size': 64, 'num_lstm_units': 128,
              'lstm_dropout_keep_prob': 0.35, 'padded_len': 60, 'num_tag': 5}
    estimate_model = BiLSTMCRF(config,test_mode=True)
    graph = tf.Graph()
    lines = open(datapath,'r').readlines()
    max_len = 60
    if polyglot:
        em_weights = embedding_from_POLYGLOT()
    else:
        em_weights = embbeding_from_scratch()
    with graph.as_default():
        print(len(vocab._vocab))
        estimate_model.build(em_weights)
        unpadded_input, seq_length, predict_tag = estimate_model.unpadded_input, estimate_model.seq_length, estimate_model.predict_tag
        saver = tf.train.Saver()
        with tf.Session(graph=graph) as session:
            saver.restore(session,modelpath)
            for line in lines:
                line = ''.join(line.decode('utf-8').strip().split())
                tags = []
                full_len = 0
                for batch,seq_len in seq2batches(line,vocab,60):
                    feed_dict = {unpadded_input: batch,
                                 seq_length: seq_len}
                    pred = session.run(predict_tag,feed_dict=feed_dict)
                    tags.extend(pred[0].tolist())
                    full_len += seq_len
                line = line[:full_len]
                tags = tags[:full_len]
                partitioned = partition(line, tags)
                output.write(''.join(partitioned).encode('utf-8')+'\n')
    output.close()

def partition(seqs,tags):
    id2tag = 'USBEM'
    partitioned = []
    assert len(seqs)==len(tags)
    for i in range(len(seqs)):
        t = id2tag[tags[i]]
        if t == 'B':
            partitioned.append(seqs[i])
        if t == 'M':
            partitioned.append(seqs[i])
        if t == 'E':
            partitioned.append(seqs[i])
            partitioned.append('/')
        if t == 'S':
            partitioned.append(seqs[i])
            partitioned.append('/')
        if t == 'U':
            partitioned.append(seq[i])
    return partitioned


def evaluate(modelpath,datapath,vocab,polyglot=True):
    config = {'batch_size': 1, 'embedding_size': 64, 'num_lstm_units': 128,
              'lstm_dropout_keep_prob': 0.35, 'padded_len': 60, 'num_tag': 5}
    estimate_model = BiLSTMCRF(config,test_mode=True)
    graph = tf.Graph()
    max_len = 60
    if polyglot:
        em_weights = embedding_from_POLYGLOT()
    else:
        em_weights = embbeding_from_scratch()
    with graph.as_default():
        print(len(vocab._vocab))
        estimate_model.build(em_weights)
        unpadded_input, seq_length, predict_tag = estimate_model.unpadded_input, estimate_model.seq_length, estimate_model.predict_tag
        saver = tf.train.Saver()
        def _parse_wrapper(l):
                return parse_example_queue(l)
        dataset = tf.data.TFRecordDataset(datapath).map(
            _parse_wrapper)
        with tf.name_scope('eval_inputs'):
            dataset = dataset.shuffle(
                    buffer_size=256).repeat(1).shuffle(buffer_size=256)
            iterator = dataset.make_one_shot_iterator()
            item = iterator.get_next()
        with tf.Session(graph=graph) as session:
            saver.restore(session,modelpath)
            pred = []
            gt = []
            print('evaluating performance,please wait...')
            try:
                while True:
                    char_id_seqs, tag_seqs, sequence_length = session.run(item)
                    feed_dict = {unpadded_input: char_id_seqs,
                                 seq_length: sequence_length}
                    predicted = session.run(predict_tag,feed_dict=feed_dict)
                    predicted = predicted[0][:sequence_length].tolist()
                    # print(sequence_length,' '.join(map(str,char_id_seqs)),''.join(str(tag_seqs)),predicted)
                    # print(type(tag_seqs),type(predicted))
                    gt.extend(tag_seqs.tolist())
                    pred.extend(predicted)
            except tf.errors.OutOfRangeError:
                pass
            pred = map(id_to_tag,pred)
            gt = map(id_to_tag,gt)
            eval_performance(gt,pred)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model_path', help='model path.',default=os.path.join('output/model', 'estimator.ckpt'))
    parser.add_argument('-t', '--testdata',
                        help='test data without ground truth.')
    parser.add_argument('-e','--evaldata',help='evaluation data with ground truth.')
    parser.add_argument('-p','--polyglot',help='use PLOYGLOT embedding.',action='store_true')
    opt = parser.parse_args()
    dictionary = pickle.load(open('./output/dictionary.pkl', 'rb'))
    vocab = Vocabulary(dictionary)
    if opt.testdata:
        test(opt.model_path, opt.testdata,vocab,opt.polyglot)
    if opt.evaldata:
        evaluate(opt.model_path, opt.evaldata,vocab,opt.polyglot)

if __name__ == '__main__':
    main()
