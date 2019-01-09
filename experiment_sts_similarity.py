from data_reader import *
from similarity_ideas import *


if __name__ == '__main__':
    texts, sims = load_dev_texts_and_similarities()

    # lsa similarity
    texts_flatten = [txt for i in range(len(texts)) for txt in texts[i]]
    """
    lsa_sim = lsa_sim(texts_flatten) # sim for all texts, we're only interested in the original sims
    lsa_sim = np.array([lsa_sim[2*i, 2*i+1] for i in range(len(texts))])
    lsa_sim = lsa_sim*5
    with open('storage/predictions/lsa_sim.txt', 'w') as f:
        for i in range(len(lsa_sim)):
            f.write(str(lsa_sim[i])+'\n')

    tfidf_sim = tf_idf_similarity(texts_flatten)
    tfidf_sim = np.array([tfidf_sim[2*i, 2*i+1] for i in range(len(texts))])
    tfidf_sim = tfidf_sim*5
    with open('storage/predictions/tfidf_sim.txt', 'w') as f:
        for i in range(len(tfidf_sim)):
            f.write(str(tfidf_sim[i])+'\n')
    """
    binary_sim = binary_cosine_sim_2(texts_flatten)
    binary_sim = np.array([binary_sim[2*i, 2*i+1] for i in range(len(texts))])
    binary_sim = binary_sim*5
    with open('storage/predictions/binary_sim_2.txt', 'w') as f:
        for i in range(len(binary_sim)):
            f.write(str(binary_sim[i])+'\n')
    """
    word2vec_similarities = word2vec_sim(texts_flatten, True, 'cosine')
    word2vec_similarities = np.array([word2vec_similarities[2*i, 2*i+1] for i in range(len(texts))])
    word2vec_similarities = word2vec_similarities*5
    with open('storage/predictions/tfidf_weighted_word2vec_sim_cosine.txt', 'w') as f:
        for i in range(len(word2vec_similarities)):
            f.write(str(word2vec_similarities[i])+'\n')
    
    word2vec_similarities = word2vec_sim(texts_flatten, False, 'cosine')
    word2vec_similarities = np.array([word2vec_similarities[2*i, 2*i+1] for i in range(len(texts))])
    word2vec_similarities = word2vec_similarities*5
    with open('storage/predictions/avg_wv_word2vec_sim_cosine.txt', 'w') as f:
        for i in range(len(word2vec_similarities)):
            f.write(str(word2vec_similarities[i])+'\n')
    
    word2vec_similarities = word2vec_sim(texts_flatten, True, 'l2')
    word2vec_similarities = np.array([word2vec_similarities[2*i, 2*i+1] for i in range(len(texts))])
    word2vec_similarities = word2vec_similarities*5
    with open('storage/predictions/tfidf_weighted_word2vec_sim_l2.txt', 'w') as f:
        for i in range(len(word2vec_similarities)):
            f.write(str(word2vec_similarities[i])+'\n')
    
    word2vec_similarities = word2vec_sim(texts_flatten, False, 'l2')
    word2vec_similarities = np.array([word2vec_similarities[2*i, 2*i+1] for i in range(len(texts))])
    word2vec_similarities = word2vec_similarities*5
    with open('storage/predictions/avg_wv_word2vec_sim_l2.txt', 'w') as f:
        for i in range(len(word2vec_similarities)):
            f.write(str(word2vec_similarities[i])+'\n')
    
    fasttext_similarities = fasttext_sim(texts_flatten, True, 'cosine')
    fasttext_similarities = np.array([fasttext_similarities[2*i, 2*i+1] for i in range(len(texts))])
    fasttext_similarities = fasttext_similarities*5
    with open('storage/predictions/tfidf_weighted_fasttext_sim_cosine.txt', 'w') as f:
        for i in range(len(fasttext_similarities)):
            f.write(str(fasttext_similarities[i])+'\n')

    fasttext_similarities = fasttext_sim(texts_flatten, False, 'cosine')
    fasttext_similarities = np.array([fasttext_similarities[2*i, 2*i+1] for i in range(len(texts))])
    fasttext_similarities = fasttext_similarities*5
    with open('storage/predictions/avg_wv_fasttext_sim_cosine.txt', 'w') as f:
        for i in range(len(fasttext_similarities)):
            f.write(str(fasttext_similarities[i])+'\n')
    
    fasttext_similarities = fasttext_sim(texts_flatten, True, 'l2')
    fasttext_similarities = np.array([fasttext_similarities[2*i, 2*i+1] for i in range(len(texts))])
    fasttext_similarities = fasttext_similarities*5
    with open('storage/predictions/tfidf_weighted_fasttext_sim_l2.txt', 'w') as f:
        for i in range(len(fasttext_similarities)):
            f.write(str(fasttext_similarities[i])+'\n')

    fasttext_similarities = fasttext_sim(texts_flatten, False, 'l2')
    fasttext_similarities = np.array([fasttext_similarities[2*i, 2*i+1] for i in range(len(texts))])
    fasttext_similarities = fasttext_similarities*5
    with open('storage/predictions/avg_wv_fasttext_sim_l2.txt', 'w') as f:
        for i in range(len(fasttext_similarities)):
            f.write(str(fasttext_similarities[i])+'\n')
    
    glove_similarities = glove_sim(texts_flatten, True, 'cosine')
    glove_similarities = np.array([glove_similarities[2*i, 2*i+1] for i in range(len(texts))])
    glove_similarities = glove_similarities*5
    with open('storage/predictions/tfidf_weighted_glove_sim_cosine.txt', 'w') as f:
        for i in range(len(glove_similarities)):
            f.write(str(glove_similarities[i])+'\n')

    glove_similarities = glove_sim(texts_flatten, False, 'cosine')
    glove_similarities = np.array([glove_similarities[2*i, 2*i+1] for i in range(len(texts))])
    glove_similarities = glove_similarities*5
    with open('storage/predictions/avg_wv_glove_sim_cosine.txt', 'w') as f:
        for i in range(len(glove_similarities)):
            f.write(str(glove_similarities[i])+'\n')
    
    glove_similarities = glove_sim(texts_flatten, True, 'l2')
    glove_similarities = np.array([glove_similarities[2*i, 2*i+1] for i in range(len(texts))])
    glove_similarities = glove_similarities*5
    with open('storage/predictions/tfidf_weighted_glove_sim_l2.txt', 'w') as f:
        for i in range(len(glove_similarities)):
            f.write(str(glove_similarities[i])+'\n')

    glove_similarities = glove_sim(texts_flatten, False, 'l2')
    glove_similarities = np.array([glove_similarities[2*i, 2*i+1] for i in range(len(texts))])
    glove_similarities = glove_similarities*5
    with open('storage/predictions/avg_wv_glove_sim_l2.txt', 'w') as f:
        for i in range(len(glove_similarities)):
            f.write(str(glove_similarities[i])+'\n')

    dan_sim = dan_sim(texts_flatten)
    dan_sim = np.array([dan_sim[2*i, 2*i+1] for i in range(len(texts))])
    dan_sim = dan_sim*5
    with open('storage/predictions/dan_sim.txt', 'w') as f:
        for i in range(len(dan_sim)):
            f.write(str(dan_sim[i])+'\n')
    """
