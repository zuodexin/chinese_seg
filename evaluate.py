import CRFPP
import argparse


def evaluate(tagger,path):
    tagger.clear()
    fp = open(path,'r')
    lines = fp.readlines()
    n_col = len(lines[0])
    n_row = len(lines)
    gt = []
    for i in xrange(n_row):
        cols = lines[i].split()
        tag = cols[-1]
        observe = ' '.join(cols[:-2])
        gt.append(tag)
        tagger.add(observe)
    # print n_row
    print "column size: " , tagger.xsize()
    print "token size: " , tagger.size()
    print "tag size: " , tagger.ysize()
    print "tagset information:"
    ysize = tagger.ysize()
    for i in range(0, ysize):
        print "tag " , i , " " , tagger.yname(i)
    # parse and change internal stated as 'parsed'
    tagger.parse()
    pred = [ tagger.y2(i) for i in xrange(len(gt))]
    print "conditional prob=" , tagger.prob(), " log(Z)=" , tagger.Z()
    #    print "Details",
    #    for j in range(0, (ysize-1)):
    #       print "\t" , tagger.yname(j) , "/prob=" , tagger.prob(i,j),"/alpha=" , tagger.alpha(i, j),"/beta=" , tagger.beta(i, j),
    #    print "\n",
    eval_performance(gt,pred)
    fp.close()

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
    print 'num words of predicted: ',wc_pred
    print 'num words of ground truth: ',wc_gt
    print 'num words of correct: ',wc_correct
    print "P = %f, R = %f, F-score = %f" % (P, R, (2*P*R)/(P+R))


def test(tagger,path):
    fp = open(path,'r')
    output = open('output/result.txt','w')
    lines = fp.readlines()
    n_line = len(lines)
    for i,line in enumerate(lines):
        tagger.clear()
        line = line.strip('\n\t').decode('utf-8')
        tokens = [line[j].encode('utf-8') for j in xrange(len(line)) if line[j]!=' ']
        print len(line),len(tokens)
        for t in tokens:
            tagger.add(t)
        print "column size: " , tagger.xsize()
        print "token size: " , tagger.size()
        print "tag size: " , tagger.ysize()
        print "tagset information:"
        ysize = tagger.ysize()
        for i in range(0, ysize):
            print "tag " , i , " " , tagger.yname(i)
        tagger.parse()
        print "conditional prob=" , tagger.prob(), " log(Z)=" , tagger.Z()
        size = tagger.size()
        xsize = tagger.xsize()
        for i in range(size):
            t = tagger.y2(i)
            if t == 'B':
                output.write(tokens[i])
            if t == 'M':
                output.write(tokens[i])
            if t == 'E':
                output.write(tokens[i])
                output.write('/')
            if t == 'S':
                output.write(tokens[i])
                output.write('/')
        output.write('\n')
    output.close()
    fp.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',help='model path.')
    parser.add_argument('-n','--num',help='top n result.')
    parser.add_argument('-e','--evaldata',help='evaluation data with ground truth.')
    parser.add_argument('-t','--testdata',help='test data without ground truth.')
    opt = parser.parse_args()
    tagger = CRFPP.Tagger("-m {} -n {} ".format(opt.model,opt.num))
    if opt.testdata:
        test(tagger,opt.testdata)
    if opt.evaldata:
        evaluate(tagger,opt.evaldata)


if __name__ == '__main__':
    main()