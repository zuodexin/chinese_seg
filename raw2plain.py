import re


def convert_PKU1998_01():
    raw_fp = open('datasets/PKU1998_01/data.txt', 'r')
    plain_fp = open('output/data/plain.txt', 'w')

    lines = raw_fp.readlines()
    for i, line in enumerate(lines):
        line = line.decode('gbk')
        words = line.strip('\r\n\t').split()
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
            print w
            plain_fp.write(w.encode('utf-8'))
        plain_fp.write('\n')
    raw_fp.close()
    plain_fp.close()


def main():
    convert_PKU1998_01()


if __name__ == '__main__':
    main()
