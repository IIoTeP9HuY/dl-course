import numpy as np
import sklearn as sk
import pandas as pd

f = open('data/skipgram_100.model')
v, d = map(int, f.readline().split())

vectors = np.zeros((v, d), dtype=np.float32)
word2id = dict()
id2word = []

for v in xrange(v):
    line = f.readline().split()
    word2id[line[0]] = v
    id2word.append(line[0])
    vectors[v] = map(float, line[1:])

f.close()

data = pd.read_csv('data/city.csv')
