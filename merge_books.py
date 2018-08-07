import  argparse
import os
import os.path as osp


def merge(folder):
    assert osp.isdir(folder)
    output = osp.join('output','merged.txt')
    ofp = open(output,'w')
    books = os.listdir(folder)    
    for i,book in enumerate(books):
        fp = open(osp.join(folder,book),'r')
        lines = fp.readlines()
        if len(lines)<=0:
            continue
        print lines[0].decode('utf-8')
        for line in lines:
            line = line.strip()
            if(len(line)>0):
                ofp.write(line)
                ofp.write('\n')
        fp.close()
    ofp.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder',type=str,help='folder of books')
    opt = parser.parse_args()
    merge(opt.folder)

if __name__ == '__main__':
    main()