import numpy as np
from data_reader import *
from load_wv import *
from sklearn.feature_extraction.text import TfidfVectorizer


def embed_texts(texts, emb):
    emb_texts = np.zeros((len(texts), 50))
    for i, text in enumerate(texts):
        words = text.split(' ')
        for word in words:
            if word in emb.keys():
                emb_texts[i] += emb[word]
    return emb_texts

def embed_texts_tfidf_weighting(texts, emb):
    vectorizer = TfidfVectorizer()
    transformed = vectorizer.fit_transform(raw_documents=texts)
    index_word={i[1]:i[0] for i in vectorizer.vocabulary_.items()}
    nr_words = len(index_word.keys())
    emb_texts = np.zeros((len(texts), 50))
    for i, text in enumerate(texts):
        dense_transformed = np.array(transformed[i].todense())[0]
        relevant_idxs = [idx for idx in range(nr_words) if dense_transformed[idx]!=0]
        words         = [index_word[idx]        for idx in relevant_idxs]
        tfidf_weights = [dense_transformed[idx] for idx in relevant_idxs]
        for j, word in enumerate(words):
            if word in emb.keys():
                emb_texts[i] += tfidf_weights[j]*emb[word]
    return emb_texts

def glove_sims(emb_texts):
    sim = np.zeros((len(emb_texts),len(emb_texts)))
    for i, text1 in enumerate(emb_texts):
        for j, text2 in enumerate(emb_texts):
            sim[i,j] = np.dot(text1, text2) / (np.linalg.norm(text1) * np.linalg.norm(text2))
    sim = (sim+1.0)/2.0
    return sim


if __name__=='__main__':
    texts     = load_chi19p_C1_complete()
    emb       = load_glove_wv('storage/wordvectors/glove.6B.50d.txt')
    emb_texts = embed_texts_tfidf_weighting(texts, emb)
    sim       = glove_sims(emb_texts)
    np.savetxt('storage/chi19p-C1-complete-sims-tfidf.txt', sim)
    
