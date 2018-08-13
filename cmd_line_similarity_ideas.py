import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
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
    print('loading embedding, can take one minute')
    emb = word2vec.KeyedVectors.load_word2vec_format('storage/wordvectors/GoogleNews-vectors-negative300.bin', binary=True)
    print('done loading embedding')

    # remove unknown words
    wv = emb.wv
    x = [[w for w in text if w in wv] for text in x]
    for i in range(len(x)):
        if len(x[i]) == 0:
            x[i] = ['from']

    print('creating embedded texts')
    embedded_x = np.zeros((len(x), emb['from'].shape[0]), dtype=np.float32)
    print('embedding texts', len(embedded_x))
    for i in range(len(embedded_x)):
        if i % 10 ** 4 == 0: print('i: ', i, end=' ', flush=True)
        for w in x[i]: embedded_x[i] += emb[w]
        embedded_x[i] /= len(x[i])

    print('calculating distances')
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

    script              = sys.argv[0]
    similarity_tech     = sys.argv[1] # tfidf, word2vec or lsa
    results_name        = sys.argv[2] # where to store the similarity matrix

    # load the data
    ideas = study_ideas_dataset()

    if similarity_tech=='tfidf':
        pairwise_sim = tf_idf_similarity(ideas)
        np.save('storage/results/tfidf_similarity_'+results_name+'.npy', pairwise_sim)
    elif classifier_tech=='word2vec':
        pairwise_sim = word2vec_sim(ideas)
        np.save('storage/results/word2vec_similarity_'+results_name+'.npy', pairwise_sim)
    elif classifier_tech=='lsa':
        pairwise_sim = lsa_sim(ideas)
        np.save('storage/results/lsa_similarity_'+results_name+'.npy', pairwise_sim)
    else:
        print('This similarity technique is not offered')
