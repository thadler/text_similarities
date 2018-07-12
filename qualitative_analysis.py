import csv
import numpy as np
import scipy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

import gensim.models.keyedvectors as word2vec

from data_reader import study_ideas_dataset

if __name__ == '__main__':

    x = study_ideas_dataset()

    pairwise_sim = np.load('tfidf_similarity.npy')

    sim_1 = pairwise_sim[0, :]
    for i in range(len(x)):
        if sim_1[i] >= 0.2:
            print(x[i], sim_1[i])

    # quit()

    print('TFIDF EXAMPLES:')
    temp_max = 0.0
    temp_avg = 0.5
    temp_min = 1.0
    for i in range(len(pairwise_sim)):
        for j in range(len(pairwise_sim)):
            if i == j: continue
            if pairwise_sim[i, j] > temp_max and pairwise_sim[i, j] < 0.999:
                temp_max = pairwise_sim[i, j]
                x_max_idx, y_max_idx = i, j
            if pairwise_sim[i, j] < temp_min:
                temp_min = pairwise_sim[i, j]
                x_min_idx, y_min_idx = i, j
            if np.abs(pairwise_sim[i, j] - temp_avg) < 0.05:
                x_avg_idx, y_avg_idx = i, j

    print('Very similar')
    print(x[x_max_idx])
    print(x[y_max_idx])
    print(pairwise_sim[x_max_idx, y_max_idx])

    print('Very dissimilar')
    print(x[x_min_idx])
    print(x[y_min_idx])
    print(pairwise_sim[x_min_idx, y_min_idx])

    print('Average similarity')
    print(x[x_avg_idx])
    print(x[y_avg_idx])
    print(pairwise_sim[x_avg_idx, y_avg_idx])

    pairwise_sim = np.load('word2vec_similarity.npy')

    sim_1 = pairwise_sim[0, :]
    for i in range(len(x)):
        if sim_1[i] >= 0.85:
            print(x[i], sim_1[i])
    quit()

    print('WORD TO VEC EXAMPLES:')
    temp_max = 0.0
    temp_avg = 0.5
    temp_min = 1.0
    for i in range(len(pairwise_sim)):
        for j in range(len(pairwise_sim)):
            if i == j: continue
            if pairwise_sim[i, j] > temp_max and pairwise_sim[i, j] < 0.999:
                temp_max = pairwise_sim[i, j]
                x_max_idx, y_max_idx = i, j
            if pairwise_sim[i, j] < temp_min:
                temp_min = pairwise_sim[i, j]
                x_min_idx, y_min_idx = i, j
            if np.abs(pairwise_sim[i, j] - temp_avg) < 0.05:
                x_avg_idx, y_avg_idx = i, j

    print('Very similar')
    print(x[x_max_idx])
    print(x[y_max_idx])
    print(pairwise_sim[x_max_idx, y_max_idx])

    print('Very dissimilar')
    print(x[x_min_idx])
    print(x[y_min_idx])
    print(pairwise_sim[x_min_idx, y_min_idx])

    print('Average similarity')
    print(x[x_avg_idx])
    print(x[y_avg_idx])
    print(pairwise_sim[x_avg_idx, y_avg_idx])
