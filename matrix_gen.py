# coding=utf-8

import codecs


INPUT_FILE = 'predict.txt'
OUTPUT_FILE = 'matrix.txt'


def matrix_generate():
    with codecs.open(INPUT_FILE, 'r', 'utf-8') as f:
        data = f.readlines()

    statistics = [[0] * 18 for i in xrange(18)]

    for i in data:
        p, t = map(int, i.split('\t'))
        statistics[p][t] += 1

    with codecs.open(OUTPUT_FILE, 'w', 'utf-8') as out:
        for i in xrange(18):
            content = '\t'.join(map(str, statistics[i]))
            out.write(content+'\n')


if __name__ == '__main__':
    matrix_generate()