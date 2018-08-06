# import csv
import numpy as np
# import scipy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, pairwise_distances

import gensim.models.keyedvectors as word2vec

from data_reader import study_ideas_dataset


def tf_idf_similarity(texts):
    tfidf = TfidfVectorizer().fit_transform(texts)
    pairwise_similarity = tfidf * tfidf.T
    # automatically positive for tfidf
    print(pairwise_similarity[:5, :5])
    return pairwise_similarity.todense()


def word2vec_sim(x):
    xx = [[w.strip() for w in text.split()] for text in x]
    print('loading embedding')
    emb = word2vec.KeyedVectors.load_word2vec_format('storage/wordvectors/GoogleNews-vectors-negative300.bin',
                                                     binary=True)
    print('done loading embedding')

    # print(np.mean([l for l in map(len,x1)]), np.mean([l for l in map(len,x2)]))
    wv = emb.wv
    x = [[w for w in text if w in wv] for text in x]

    print('done changing texts')
    for i in range(len(x)):
        if len(x[i]) == 0:
            x[i] = ['from']

    print('create embedded texts')
    embedded_x = np.zeros((len(x), emb['from'].shape[0]), dtype=np.float32)
    print('embedding texts', len(embedded_x))
    for i in range(len(embedded_x)):
        if i % 10 ** 4 == 0: print('i: ', i, end=' ', flush=True)
        for w in x[i]: embedded_x[i] += emb[w]
        embedded_x[i] /= len(x[i])

    print('calculate distances')
    dist = np.zeros((len(embedded_x), len(embedded_x)))
    for i in range(len(embedded_x)):
        for j in range(len(embedded_x)):
            dist[i, j] = np.linalg.norm(embedded_x[i] - embedded_x[j])
    dist /= np.max(dist)
    sim = 1 - dist
    print(sim[:5, :5])
    return sim


def lsa_sim(texts):
    #vectorizer = TfidfVectorizer(tokenizer=Tokenizer(), stop_words='english', use_idf=True, smooth_idf=True)
    vectorizer      = TfidfVectorizer()
    print(vectorizer)
    svd_model       = TruncatedSVD(n_components=500, algorithm='randomized', n_iter=10, random_state=42)
    svd_transformer = Pipeline([('tfidf', vectorizer), ('svd', svd_model)])
    svd_matrix      = svd_transformer.fit_transform(texts)
    print(svd_matrix)

    transformed     = svd_transformer.transform(texts)
    sim = 1 - pairwise_distances(transformed, svd_matrix, metric='cosine', n_jobs=-1)
    print(sim[:5,:5])
    return sim 


if __name__ == '__main__':
    x = study_ideas_dataset()

    pairwise_sim = tf_idf_similarity(x)
    np.save('storage/results/tfidf_similarity.npy', pairwise_sim)
    pairwise_sim = word2vec_sim(x)
    np.save('storage/results/word2vec_similarity.npy', pairwise_sim)
    pairwise_sim = lsa_sim(x)
    np.save('storage/results/lsa_similarity.npy', pairwise_sim)
