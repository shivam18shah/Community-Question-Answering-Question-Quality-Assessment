import numpy as np
import pandas as pd
import os
import pickle

GLOVE_FILE = os.path.join('dataset', 'glove.6B.100d.txt')
PICKLE_PATH = os.path.join('dataset', 'glove_100.pkl')

def get_glove_model(GLOVE_FILE):
    f = open(GLOVE_FILE, encoding="utf8")
    model = dict()
    print('Getting Glove Vectors ...')
    for line in f:
        shabdo = line.split(' ')
        word = shabdo[0]
        embedding = np.array([float(i) for i in shabdo[1:]])
        model[word] = embedding
    f.close()
    return model

def main():
    glove_wordmap = get_glove_model(GLOVE_FILE)
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(glove_wordmap, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__=='__main__':
    main()
