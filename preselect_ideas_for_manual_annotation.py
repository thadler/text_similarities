import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def load_glove():
    print('loading glove embedding')
    glove = open('storage/wordvectors/glove.6B.50d.txt', 'r')
    emb = dict()
    for i, line in enumerate(glove):
        if (i+1)%10000==0: print(i+1, end=', ', flush=True)
        line = line.strip().split(' ')
        key, val = line[0], np.array([float(number) for number in line[1:]])
        emb[key] = val
    print('done loading embedding')
    return emb


def load_chi():
    texts = open('storage/datasets/chi19p-C1-complete.csv', 'r').readlines()
    texts = texts[1:] # remove header
    texts = [text.replace('"','').replace('.',' .').replace(',',' ,').split(',')[2] for text in texts]
    return texts


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
    #transformed = [' '.join(texts[i]) for i in range(len(texts))]
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
    texts = load_chi()
    emb   = load_glove()
    #emb_texts = embed_texts(texts, emb)
    emb_texts = embed_texts_tfidf_weighting(texts, emb)
    sim = glove_sims(emb_texts)
    np.savetxt('storage/chi19p-C1-complete-sims-tfidf.txt', sim)
    
