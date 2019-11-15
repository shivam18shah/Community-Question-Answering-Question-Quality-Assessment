import numpy as np
import os
import nltk
import math
import pickle
# nltk.download('punkt')
from nltk import sent_tokenize, word_tokenize
import time
st = time.time()

GLOVE_FILE = os.path.join('dataset', 'glove.6B.100d.txt')
# SIZE_OF_SENTENCE = 20
# SIZE_OF_TITLE = 8
GLOVE_SIZE = 100
PICKLE_PATH = os.path.join('dataset', 'glove_100.pkl')

def get_similarity(head, body, lim):
    # cosine is a naive choice, don't use it
    # den = np.linalg.norm(head)*np.linalg.norm(body)
    # return -0.01 if den == 0 else np.dot(head, body) / den
    scores = 0
    # print(head.shape, body.shape)
    for h in head:
        h_norm = np.linalg.norm(h)#, ord='inf')
        if h_norm == 0:
            continue
        score = []
        for b in body:
            b_norm = np.linalg.norm(b)#, ord='inf')
            if b_norm == 0:
                continue
            hb = h @ b
            # print(hb)
            score.append(hb/(h_norm*b_norm))
        # s_norm = np.linalg.norm(score)
        # print(max(score)/len(score))
        scores += max(score)/len(score)#/sum(score)
        # print(scores)
    return scores
    
    # score = np.dot(head, body.T)
    # return np.sum(score)/SIZE_OF_SENTENCE


def get_glove_model(PICKLE_PATH):
    with open(PICKLE_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

def get_embeddings(title, question, GLOVE_WORDMAP = get_glove_model(PICKLE_PATH)):
    t = word_tokenize(title)
    q = word_tokenize(question)
    vec = np.array([0]*GLOVE_SIZE)
    for word in t:
        word = word.lower()
        if GLOVE_WORDMAP.get(word) is not None:
            vec = np.vstack((vec, np.array(GLOVE_WORDMAP[word])))
        else:
            vec = np.vstack((vec, np.array([0]*GLOVE_SIZE)))
    for i in range(len(t), max(len(t),len(q))):
        vec = np.vstack((vec, np.array([0]*GLOVE_SIZE)))
    res = np.array([0]*GLOVE_SIZE)
    for word in q:
        word=word.lower()
        if GLOVE_WORDMAP.get(word) is not None:
            res = np.vstack((res, np.array(GLOVE_WORDMAP[word])))
        else:
            res = np.vstack((res, np.array([0]*GLOVE_SIZE)))
    for i in range(len(q), max(len(t),len(q))):
        res = np.vstack((res, np.array([0]*GLOVE_SIZE)))
    return vec[1:], res[1:]

def get_similarity_score(titles, questions):
    scores = np.array([])
    for title, question in zip(titles, questions):
        head, body = get_embeddings(title, question)
        similarity = get_similarity(head, body, min(len(title), len(question)))
        scores = np.append(scores, similarity)
        # print(similarity)
    return np.array(scores)

def main():
    print(time.time() - st, ' seconds to get the Glove model.')
    # TODO you code here

    # pass as arguments two list of strings titles and questions
    # returns a similarity score as a list of float with length equal to number of observations in titles and questions
    # replace the following variables respectively

    # just read csv to dataframe and uncomment the next two lines, comment the two lines after that
    # titles = df['title'].tolist()
    # questions = df['question'].tolist()
    titles = ['let us try much similar sentences now', 'Most similar sentences can be twenty words in length and both of them being exactly the same in size and words']
    questions = ['Most similar sentences can be twenty words in length and both of them being exactly the same in size and words', 'Most similar sentences can be twenty words in length and both of them being exactly the same in size and words']
    # titles = ['Most similar sentences can be twenty words in length and both of them being exactly the same in size and words']
    # questions = ['Most similar sentences can be twenty words in length and both of them being exactly the same in size and words']
    scores = get_similarity_score(titles, questions)
    print(scores)
    # add these scores as another column to the feature dataframe
    # write the df to the csv

if __name__=='__main__':
    main()