# imports
import sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.metrics import pairwise_distances
from nltk.tokenize import word_tokenize

from load_wv import *
from data_reader import *

# classic functions that compute the similarity between texts

def tf_idf_similarity(texts):
    """Embeds texts in tf idf vector representations then stores the cosine
       similarity between all texts in a similarity matrix
    Keyword arguments:
    texts -- an iterable of strings where each string represents a text
    """
    tfidf = TfidfVectorizer(use_idf=True, max_df=0.18, norm='l2').fit_transform(texts)
    pairwise_similarity = tfidf * tfidf.T
    # automatically in [0,1] for tfidf
    print(pairwise_similarity[:5, :5])
    return pairwise_similarity.todense()

def word2vec_sim(texts, tfidf_weighting=True, norm='l2'):
    """Embeds texts in tf idf vector representations then stores the cosine
       similarity between all texts in a similarity matrix
    Keyword arguments:
    texts -- an iterable of strings where each string represents a text
    tfidf_weighting -- a boolean which decides whether the word vectors are
                       weighted according to their inverse document frequency.
                       generally True is recommended.
    norm -- string, currently either l2 norm or cosine similarity
    """
    # preprocessing texts for tfidf weighting
    if tfidf_weighting:
        vectorizer = TfidfVectorizer()
        transformed = vectorizer.fit_transform(raw_documents=texts)
        index_word={i[1]:i[0] for i in vectorizer.vocabulary_.items()}
        nr_words = len(index_word.keys())
    
    # preprocess texts for embedding
    texts = [text.replace('"','').replace('.',' .').replace(',',' ,') for text in texts]
    texts = [[w.strip() for w in text.split()] for text in texts]
    wv = load_word2vec_wv('storage/wordvectors/GoogleNews-vectors-negative300.bin')
    texts = [[w for w in text if w in wv] for text in texts]
    [print('error - text without words') for text in texts if len(text)==0]

    print('creating embedded texts')
    embedded_texts = np.zeros((len(texts), wv['from'].shape[0]), dtype=np.float32)
    print('embedding texts', len(embedded_texts))
    for i, text in enumerate(texts):
        if i % 10 ** 4 == 0: print('i: ', i, end=' ', flush=True)
        if tfidf_weighting:
            dense_transformed = np.array(transformed[i].todense())[0]
            relevant_idxs = [idx for idx in range(nr_words) if dense_transformed[idx]!=0]
            words         = [index_word[idx]        for idx in relevant_idxs]
            tfidf_weights = [dense_transformed[idx] for idx in relevant_idxs]
            for j, w in enumerate(words):
                if w in wv.vocab:
                    embedded_texts[i] += wv[w]*tfidf_weights[j]
        else:
            for w in text: embedded_texts[i] += wv[w]
        embedded_texts[i] /= len(text)

    print('calculating similarities')
    if norm=='l2':
        dist = np.zeros((len(embedded_texts), len(embedded_texts)))
        for i in range(len(embedded_texts)):
            for j in range(len(embedded_texts)):
                dist[i, j] = np.linalg.norm(embedded_texts[i] - embedded_texts[j])
        dist /= np.max(dist)
        sim = 1 - dist
    elif norm=='cosine':
        sim = embedded_texts.dot(embedded_texts.T)
        for i in range(len(sim)):
            for j in range(len(sim)):
                sim[i,j] /= (np.linalg.norm(embedded_texts[i])*np.linalg.norm(embedded_texts[j]))
    return sim

def fasttext_sim(texts, tfidf_weighting=True, norm='l2'):
    """Embeds texts in tf idf vector representations then stores the cosine
       similarity between all texts in a similarity matrix
    Keyword arguments:
    texts -- an iterable of strings where each string represents a text
    tfidf_weighting -- a boolean which decides whether the word vectors are
                       weighted according to their inverse document frequency.
                       generally True is recommended.
    norm -- string, currently either l2 norm or cosine similarity
    """
    # preprocessing texts for tfidf weighting
    if tfidf_weighting:
        vectorizer = TfidfVectorizer()
        transformed = vectorizer.fit_transform(raw_documents=texts)
        index_word={i[1]:i[0] for i in vectorizer.vocabulary_.items()}
        nr_words = len(index_word.keys())

    # preprocess texts for embedding
    texts = [text.replace('"','').replace('.',' .').replace(',',' ,') for text in texts]
    texts = [[w.strip() for w in text.split()] for text in texts]
    wv    = load_fasttext_wv('storage/wordvectors/crawl-300d-2M-subword.vec')
    texts = [[w for w in text if w in wv] for text in texts]
    [print('error - text without words') for text in texts if len(text)==0]

    print('creating text embeddings')
    embedded_texts = np.zeros((len(texts), wv['from'].shape[0]), dtype=np.float32)
    print('embedding ', len(embedded_texts), ' texts')
    for i, text in enumerate(texts):
        if (i+1)%10**4==0: print(i+1, end=' ', flush=True)
        if tfidf_weighting:
            dense_transformed = np.array(transformed[i].todense())[0]
            relevant_idxs = [idx for idx in range(nr_words) if dense_transformed[idx]!=0]
            words         = [index_word[idx]        for idx in relevant_idxs]
            tfidf_weights = [dense_transformed[idx] for idx in relevant_idxs]
            for j, w in enumerate(words):
                if w in wv.keys():
                    embedded_texts[i] += wv[w]*tfidf_weights[j]
        else:
            for w in text: embedded_texts[i] += wv[w]
        embedded_texts[i] /= len(text)

    print('calculating similarities')
    if norm=='l2':
        dist = np.zeros((len(embedded_texts), len(embedded_texts)))
        for i in range(len(embedded_texts)):
            for j in range(len(embedded_texts)):
                dist[i, j] = np.linalg.norm(embedded_texts[i] - embedded_texts[j])
        dist /= np.max(dist)
        sim = 1 - dist
    elif norm=='cosine':
        sim = embedded_texts.dot(embedded_texts.T)
        for i in range(len(sim)):
            for j in range(len(sim)):
                sim[i,j] /= (np.linalg.norm(embedded_texts[i])*np.linalg.norm(embedded_texts[j]))
    return sim

def glove_sim(texts, tfidf_weighting=True, norm='l2'):
    """Embeds texts in tf idf vector representations then stores the cosine
       similarity between all texts in a similarity matrix
    Keyword arguments:
    texts -- an iterable of strings where each string represents a text
    tfidf_weighting -- a boolean which decides whether the word vectors are
                       weighted according to their inverse document frequency.
                       generally True is recommended.
    norm -- string, currently either l2 norm or cosine similarity
    """
    # preprocessing texts for tfidf weighting
    if tfidf_weighting:
        vectorizer = TfidfVectorizer()
        transformed = vectorizer.fit_transform(raw_documents=texts)
        index_word={i[1]:i[0] for i in vectorizer.vocabulary_.items()}
        nr_words = len(index_word.keys())

    # preprocessing texts for embedding
    texts = [text.replace('"','').replace('.',' .').replace(',',' ,') for text in texts]
    texts = [[w.strip() for w in text.split()] for text in texts]
    wv = load_glove_wv('storage/wordvectors/glove.840B.300d.txt')
    texts = [[w for w in text if w in wv.keys()] for text in texts]
    [print('error - text without words') for text in texts if len(text)==0]
    
    print('creating text embeddings')
    embedded_texts = np.zeros((len(texts), wv['from'].shape[0]), dtype=np.float32)
    print('embedding texts', len(embedded_texts))
    for i, text in enumerate(texts):
        if (i+1)%10**4==0: print(i, end=' ', flush=True)
        if tfidf_weighting:
            dense_transformed = np.array(transformed[i].todense())[0]
            relevant_idxs = [idx for idx in range(nr_words) if dense_transformed[idx]!=0]
            words         = [index_word[idx]        for idx in relevant_idxs]
            tfidf_weights = [dense_transformed[idx] for idx in relevant_idxs]
            for j, w in enumerate(words):
                if w in wv.keys():
                    embedded_texts[i] += wv[w]*tfidf_weights[j]
        else: 
            for w in text: embedded_texts[i] += wv[w]
        embedded_texts[i] /= len(text)

    print('calculating similarities')
    if norm=='l2':
        dist = np.zeros((len(embedded_texts), len(embedded_texts)))
        for i in range(len(embedded_texts)):
            for j in range(len(embedded_texts)):
                dist[i, j] = np.linalg.norm(embedded_texts[i] - embedded_texts[j])
        dist /= np.max(dist)
        sim = 1 - dist
    elif norm=='cosine':
        sim = embedded_texts.dot(embedded_texts.T)
        for i in range(len(sim)):
            for j in range(len(sim)):
                sim[i,j] /= (np.linalg.norm(embedded_texts[i])*np.linalg.norm(embedded_texts[j]))
    return sim

def lsa_sim(texts):
    """Embeds texts in lsa-representations then stores the cosine similarity
    between all texts in a similarity matrix
    Keyword arguments:
    texts -- an iterable of strings where each string represents a text
    """
    vectorizer      = TfidfVectorizer()
    # why 500? scikit-learn recommends: For LSA, a value of 100 is recommended.
    svd_model       = TruncatedSVD(n_components=500, algorithm='randomized', n_iter=10, random_state=42)
    svd_transformer = Pipeline([('tfidf', vectorizer), ('svd', svd_model)])
    svd_matrix      = svd_transformer.fit_transform(texts)
    transformed    = svd_transformer.transform(texts)
    pairwise_dists = pairwise_distances(transformed, svd_matrix, metric='cosine', n_jobs=-1)
    pairwise_dists = pairwise_dists - np.min(pairwise_dists)
    pairwise_dists = pairwise_dists / np.max(pairwise_dists)
    sim            = 1 - pairwise_dists
    return sim

def binary_cosine_sim(texts):
    all_texts = " ".join(texts)
    words = set(word_tokenize(all_texts, language='english'))
    words2idx = {word:i for i, word in enumerate(words)}
    binary_texts = np.zeros((len(texts), len(words2idx)))
    texts = [word_tokenize(t, language='english') for t in texts]
    for i, text in enumerate(texts):
        for w in text:
            binary_texts[i, words2idx[w]] = 1
    print('calculating similarities')
    sim = np.zeros((len(texts), len(texts)))
    for i in range(0, len(binary_texts), 2):
        sim[i, i+1] = np.dot(binary_texts[i], binary_texts[i+1]) / (np.linalg.norm(binary_texts[i]) * np.linalg.norm(binary_texts[i+1]))
    return sim

def dan_sim(texts):
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        text_embeddings = np.array(session.run(embed(texts)))
        pred_sims = text_embeddings.dot(text_embeddings.T)
    return pred_sims


if __name__ == '__main__':
    script              = sys.argv[0]
    similarity_tech     = sys.argv[1] # tfidf, word2vec or lsa
    results_name        = sys.argv[2] # where to store the similarity matrix

    # load the data
    #ideas = study_ideas_dataset('storage/datasets/chi19s1-study-ideas.csv')
    ideas, _ = load_chi19ppre_partial_approved()

    if similarity_tech=='tfidf':
        pairwise_sim = tf_idf_similarity(ideas)
        np.savetxt('storage/predictions/tfidf_similarity_'+results_name+'.txt', pairwise_sim)
    elif similarity_tech=='word2vec':
        pairwise_sim = word2vec_sim(ideas)
        np.savetxt('storage/predictions/word2vec_similarity_'+results_name+'.txt', pairwise_sim)
    elif similarity_tech=='lsa':
        pairwise_sim = lsa_sim(ideas)
        np.savetxt('storage/predictions/lsa_similarity_'+results_name+'.txt', pairwise_sim)
    elif similarity_tech=='binary':
        pairwise_sim = lsa_sim(ideas)
        np.savetxt('storage/predictions/binary_similarity_'+results_name+'.txt', pairwise_sim)
    elif similarity_tech=='glove':
        pairwise_sim = glove_sim(ideas)
        np.savetxt('storage/predictions/glove_similarity_'+results_name+'.txt', pairwise_sim)
    elif similarity_tech=='dan':
        pairwise_sim = glove_sim(ideas)
        np.savetxt('storage/predictions/dan_similarity_'+results_name+'.txt', pairwise_sim)
    else:
        print('This similarity technique is not offered')
