import io
import numpy as np
import gensim.models.keyedvectors as word2vec

def load_word2vec_wv(fname):
    print('loading embedding, can take one minute')
    emb = word2vec.KeyedVectors.load_word2vec_format(fname, binary=True)
    wv = emb.wv
    print('done loading embedding')
    return wv

def load_fasttext_wv(fname): #2-3min loading time
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    wv = {}
    print('loading fasttext embedding. Embeds 2.000.000 words. Can take two minutes.')
    for i, line in enumerate(fin):
        tokens = line.rstrip().split(' ')
        wv[tokens[0]] = np.array([float(token) for token in tokens[1:]])
        if (i+1)%10000==0: print(i+1, end=', ', flush=True)
    print('done loading fasttext embedding')
    return wv

def load_glove_wv(fname):
    print('loading glove embedding, can take one minute')
    glove = open(fname, 'r')
    wv = dict()
    for i, line in enumerate(glove):
        if (i+1)%10000==0: print(i+1, end=', ', flush=True)
        line = line.strip().split(' ')
        key, val = line[0], np.array([float(number) for number in line[1:]])
        wv[key] = val
    print('done loading glove embedding')
    return wv


if __name__=='__main__':
    word2vec_wv = load_word2vec_wv('storage/wordvectors/GoogleNews-vectors-negative300.bin')
    print(word2vec_wv['hi'].shape)
    ft_wv = load_fasttext_wv('storage/wordvectors/crawl-300d-2M-subword.vec')
    print(ft_wv['hi'].shape)
    glove_wv = load_glove_wv('storage/wordvectors/glove.840B.300d.txt')
    print(glove_wv['hi'].shape)

    
