import bcolz
import numpy as np
import pickle

glove_path = '.'
infile = 'glove.6B.300d.txt'
outfile = '6B.300'
dim = 300
vocab_size = 400000

words = []
idx = 0
word2idx = {}
vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/{outfile}.dat', mode='w')

with open(f'{glove_path}/{infile}', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)
    
vectors = bcolz.carray(vectors[1:].reshape((vocab_size, dim)), rootdir=f'{glove_path}/{outfile}.dat', mode='w')
vectors.flush()
pickle.dump(words, open(f'{glove_path}/{outfile}_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'{glove_path}/{outfile}_idx.pkl', 'wb'))
