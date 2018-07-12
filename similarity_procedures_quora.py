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

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim.models.keyedvectors as word2vec

from data_reader import quora_duplicate_questions_dataset


# test function
def classification(x1_train, x2_train, x1_test, x2_test, y_train, y_test):
    count_vect = CountVectorizer().fit(x1_train + x2_train)
    x1_train = count_vect.transform(x1_train)
    x2_train = count_vect.transform(x2_train)
    x1_test = count_vect.transform(x1_test)
    x2_test = count_vect.transform(x2_test)

    tf_transformer = TfidfTransformer(use_idf=True).fit(x1_train + x2_train)
    x1_train = tf_transformer.transform(x1_train)
    x2_train = tf_transformer.transform(x2_train)
    x1_test = tf_transformer.transform(x1_test)
    x2_test = tf_transformer.transform(x2_test)

    x_train = scipy.sparse.hstack((x1_train, x2_train))
    x_test = scipy.sparse.hstack((x1_test, x2_test))

    print(x_train.shape)
    print(x_test.shape)

    for clf in [MultinomialNB(), SGDClassifier()]:
        clf = clf.fit(x_train, y_train)
        predicted_train = clf.predict(x_train)
        print('training accuracy: ', np.mean(predicted_train == y_train))
        predicted_test = clf.predict(x_test)
        print('testing accuracy: ', np.mean(predicted_test == y_test))


def tfids_sim(x1, x2, y):
    tfidf = TfidfVectorizer().fit(x1 + x2)
    similarities = np.zeros((len(x1), 1))
    for i in range(len(y)):
        if i % 10000 == 0 and i > 1: print(i, ' of ', len(y)); break
        tf_x1, tf_x2 = tfidf.transform([x1[i]]), tfidf.transform([x2[i]])
        similarities[i, 0] = (tf_x1 * tf_x2.T)[0, 0]  # due to sparse matrix notation
    sim_y0 = np.mean([similarities[i][0] for i in range(len(y)) if y[i] == 0])
    sim_y1 = np.mean([similarities[i][0] for i in range(len(y)) if y[i] == 1])
    print('similarity of 0 cases: ', sim_y0)
    print('similarity of 1 cases: ', sim_y1)
    clf = LinearSVC()
    clf.fit(similarities[0:int(len(y) / 2)], y[0:int(len(y) / 2)])
    pred = clf.predict(similarities[int(len(y) / 2):])
    print('accuracy: ', accuracy_score(y[int(len(y) / 2):], pred))


def word2vec_sim(x1, x2, y):
    x1 = [[w.strip() for w in text.split()] for text in x1]
    x2 = [[w.strip() for w in text.split()] for text in x2]
    print('loading embedding')
    emb = word2vec.KeyedVectors.load_word2vec_format('~/Desktop/GoogleNews-vectors-negative300.bin', binary=True)
    print('done loading embedding')

    # print(np.mean([l for l in map(len,x1)]), np.mean([l for l in map(len,x2)]))
    wv = emb.wv
    x1 = [[w for w in text if w in wv] for text in x1]
    x2 = [[w for w in text if w in wv] for text in x2]
    # print(np.mean([l for l in map(len,x1)]), np.mean([l for l in map(len,x2)]))

    print('done changing texts')
    for i in range(len(x1)):
        if len(x1[i]) == 0:
            x1[i] = ['from']
    for i in range(len(x2)):
        if len(x2[i]) == 0:
            x2[i] = ['from']

    print('create embedded texts')
    embedded_x1 = np.zeros((len(x1), emb['from'].shape[0]), dtype=np.float32)
    embedded_x2 = np.zeros((len(x2), emb['from'].shape[0]), dtype=np.float32)
    print('embedding texts', len(embedded_x1) + len(embedded_x2))
    for i in range(len(embedded_x1)):
        if i % 10 ** 4 == 0: print('i: ', i, end=' ', flush=True)
        for w in x1[i]: embedded_x1[i] += emb[w]
        embedded_x1[i] /= len(x1[i])
        for w in x2[i]: embedded_x2[i] += emb[w]
        embedded_x2[i] /= len(x2[i])

    print('create similarities')
    sim = np.zeros((len(embedded_x1), 1))
    for i in range(len(embedded_x1)):
        sim[i, 0] = np.linalg.norm(embedded_x1[i] - embedded_x2[i])

    print('classify based on similarities')
    sim_y0 = np.mean([sim[i][0] for i in range(len(y)) if y[i] == 0])
    sim_y1 = np.mean([sim[i][0] for i in range(len(y)) if y[i] == 1])
    print('distance of 0 cases: ', sim_y0)
    print('distance of 1 cases: ', sim_y1)
    clf = LinearSVC()
    clf.fit(sim[0:int(len(y) / 2)], y[0:int(len(y) / 2)])
    pred = clf.predict(sim[int(len(y) / 2):])
    print('accuracy: ', accuracy_score(y[int(len(y) / 2):], pred))

    return embedded_x1, embedded_x2


def doc2vec_similarity(x1_train, x2_train, x1_test, x2_test, y_train, y_test):
    print(len(x1_train + x2_train))
    documents = [TaggedDocument(words=txt.split(), tags='{0:06}'.format(i)) for i, txt in
                 enumerate(x1_train + x2_train)]
    print('done')
    model = Doc2Vec(documents, vector_size=100, window=4, min_count=5, workers=4)
    print(model)
    return model


if __name__ == '__main__':
    x1_train, x2_train, x1_test, x2_test, y_train, y_test = quora_duplicate_questions_dataset()
    # classification(x1_train, x2_train, x1_test, x2_test, y_train, y_test)

    # tfids_sim(x1_train+x1_test, x2_train+x2_test, y_train+y_test)
    word2vec_sim(x1_train + x1_test, x2_train + x2_test, y_train + y_test)
    # doc2vec_similarity(x1_train, x2_train, x1_test, x2_test, y_train, y_test)
