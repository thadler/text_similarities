from data_reader import *
from similarity_ideas import *


if __name__ == '__main__':
    texts, sims = load_chi19p_C1_complete()
    
    lsa_sim = lsa_sim(texts) # sim for all texts, we're only interested in the original sims
    lsa_sim = lsa_sim*5
    np.savetxt('storage/predictions/chi_c1_complete/lsa_sim.txt', lsa_sim)

    tfidf_sim = tf_idf_similarity(texts)
    tfidf_sim = tfidf_sim*5
    np.savetxt('storage/predictions/chi_c1_complete/tfidf_sim.txt', tfidf_sim)
    
    binary_sim = binary_cosine_sim(texts)
    binary_sim = binary_sim*5
    with open('storage/predictions/chi_c1_complete/binary_sim.txt', 'w') as f:
        for i in range(len(binary_sim)):
            f.write(str(binary_sim[i])+'\n')
    
    word2vec_similarities = word2vec_sim(texts, True, 'cosine')
    word2vec_similarities = word2vec_similarities*5
    np.savetxt('storage/predictions/chi_c1_complete/tfidf_weighted_word2vec_sim_cosine.txt', word2vec_similarities)
    
    word2vec_similarities = word2vec_sim(texts, False, 'cosine')
    word2vec_similarities = word2vec_similarities*5
    np.savetxt('storage/predictions/chi_c1_complete/avg_wv_word2vec_sim_cosine.txt', word2vec_similarities)
    
    word2vec_similarities = word2vec_sim(texts, True, 'l2')
    word2vec_similarities = word2vec_similarities*5
    np.savetxt('storage/predictions/chi_c1_complete/tfidf_weighted_word2vec_sim_l2.txt', word2vec_similarities)
    
    word2vec_similarities = word2vec_sim(texts, False, 'l2')
    word2vec_similarities = word2vec_similarities*5
    np.savetxt('storage/predictions/chi_c1_complete/avg_wv_word2vec_sim_l2.txt', word2vec_similarities)
    
    fasttext_similarities = fasttext_sim(texts, True, 'cosine')
    fasttext_similarities = fasttext_similarities*5
    np.savetxt('storage/predictions/chi_c1_complete/tfidf_weighted_fasttext_sim_cosine.txt', fasttext_similarities)

    fasttext_similarities = fasttext_sim(texts, False, 'cosine')
    fasttext_similarities = fasttext_similarities*5
    np.savetxt('storage/predictions/chi_c1_complete/avg_wv_fasttext_sim_cosine.txt', fasttext_similarities)
    
    fasttext_similarities = fasttext_sim(texts, True, 'l2')
    fasttext_similarities = fasttext_similarities*5
    np.savetxt('storage/predictions/chi_c1_complete/tfidf_weighted_fasttext_sim_l2.txt', fasttext_similarities)
    
    fasttext_similarities = fasttext_sim(texts, False, 'l2')
    fasttext_similarities = fasttext_similarities*5
    np.savetxt('storage/predictions/chi_c1_complete/avg_wv_fasttext_sim_l2.txt', fasttext_similarities)
    
    glove_similarities = glove_sim(texts, True, 'cosine')
    glove_similarities = glove_similarities*5
    np.savetxt('storage/predictions/chi_c1_complete/tfidf_weighted_glove_sim_cosine.txt', fasttext_similarities)

    glove_similarities = glove_sim(texts, False, 'cosine')
    glove_similarities = glove_similarities*5
    np.savetxt('storage/predictions/chi_c1_complete/avg_wv_glove_sim_cosine.txt', glove_similarities)
    
    glove_similarities = glove_sim(texts, True, 'l2')
    glove_similarities = glove_similarities*5
    np.savetxt('storage/predictions/chi_c1_complete/tfidf_weighted_glove_sim_l2.txt', glove_similarities)

    glove_similarities = glove_sim(texts, False, 'l2')
    glove_similarities = glove_similarities*5
    np.savetxt('storage/predictions/chi_c1_complete/avg_wv_glove_sim_l2.txt', glove_similarities)

    dan_sim = dan_sim(texts)
    dan_sim = dan_sim*5
    np.savetxt('storage/predictions/chi_c1_complete/dan_sim.txt', dan_sim)

