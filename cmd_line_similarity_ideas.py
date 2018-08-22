# imports
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, pairwise_distances
import gensim.models.keyedvectors as word2vec
from data_reader import study_ideas_dataset

# functions that compute the similarity between texts

def tf_idf_similarity(texts):
    """Embeds texts in tf idf vector representations then stores the cosine 
       similarity between all texts in a similarity matrix
       
    Keyword arguments:
    texts -- an iterable of strings where each string represents a text
    """
    tfidf = TfidfVectorizer().fit_transform(texts)
    pairwise_similarity = tfidf * tfidf.T
    # automatically in [0,1] for tfidf
    print(pairwise_similarity[:5, :5])
    return pairwise_similarity.todense()

def word2vec_sim(texts):
    """Embeds texts in tf idf vector representations then stores the cosine 
       similarity between all texts in a similarity matrix
       
    Keyword arguments:
    texts -- an iterable of strings where each string represents a text
    """
    texts = [[w.strip() for w in text.split()] for text in texts]
    print('loading embedding, can take one minute')
    emb = word2vec.KeyedVectors.load_word2vec_format('storage/wordvectors/GoogleNews-vectors-negative300.bin', binary=True)
    print('done loading embedding')

    # remove unknown words
    wv = emb.wv
    texts = [[w for w in text if w in wv] for text in texts]
    for i in range(len(texts)):
        if len(texts[i]) == 0:
            texts[i] = ['from']

    print('creating embedded texts')
    embedded_texts = np.zeros((len(texts), emb['from'].shape[0]), dtype=np.float32)
    print('embedding texts', len(embedded_texts))
    for i in range(len(embedded_texts)):
        if i % 10 ** 4 == 0: print('i: ', i, end=' ', flush=True)
        for w in texts[i]: embedded_texts[i] += emb[w]
        embedded_texts[i] /= len(texts[i])

    print('calculating distances')
    dist = np.zeros((len(embedded_texts), len(embedded_texts)))
    for i in range(len(embedded_texts)):
        for j in range(len(embedded_texts)):
            dist[i, j] = np.linalg.norm(embedded_texts[i] - embedded_texts[j])
    dist /= np.max(dist)
    sim = 1 - dist
    print(sim[:5, :5])
    return sim

def lsa_sim(texts):
    """Embeds texts in lsa-representations then stores the cosine similarity 
    between all texts in a similarity matrix
       
    Keyword arguments:
    texts -- an iterable of strings where each string represents a text
    """
    vectorizer      = TfidfVectorizer()
    svd_model       = TruncatedSVD(n_components=500, algorithm='randomized', n_iter=10, random_state=42)
    svd_transformer = Pipeline([('tfidf', vectorizer), ('svd', svd_model)])
    svd_matrix      = svd_transformer.fit_transform(texts)
    
    transformed    = svd_transformer.transform(texts)
    pairwise_dists = pairwise_distances(transformed, svd_matrix, metric='cosine', n_jobs=-1)
    pairwise_dists = pairwise_dists - np.min(pairwise_dists)
    pairwise_dists = pairwise_dists / np.max(pairwise_dists)
    sim            = 1 - pairwise_dists
    return sim 


if __name__ == '__main__':

    script              = sys.argv[0]
    similarity_tech     = sys.argv[1] # tfidf, word2vec or lsa
    results_name        = sys.argv[2] # where to store the similarity matrix

    # load the data
    ideas = study_ideas_dataset()

    if similarity_tech=='tfidf':
        pairwise_sim = tf_idf_similarity(ideas)
        np.savetxt('storage/results/tfidf_similarity_'+results_name+'.csv', pairwise_sim)
    elif similarity_tech=='word2vec':
        pairwise_sim = word2vec_sim(ideas)
        np.savetxt('storage/results/word2vec_similarity_'+results_name+'.csv', pairwise_sim)
    elif similarity_tech=='lsa':
        pairwise_sim = lsa_sim(ideas)
        np.savetxt('storage/results/lsa_similarity_'+results_name+'.csv', pairwise_sim)
    else:
        print('This similarity technique is not offered')
