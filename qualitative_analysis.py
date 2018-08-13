# imports
import numpy as np
import gensim.models.keyedvectors as word2vec
from data_reader import study_ideas_dataset


def load_and_show_similar_to_first_example(texts, matrix_filename):
    texts        = study_ideas_dataset()
    pairwise_sim = np.load('storage/results/'+matrix_filename)
    sim_tech     = matrix_filename.split('_')[0]

    if sim_tech=='lsa':
        print('\nSimilarity for LSA')
        sim = pairwise_sim[0, :]
        for i in range(len(texts)):
            if sim[i] >= 0.28:
                print(texts[i], sim[i])
    
    if sim_tech=='tfidf':
        print('\nSimilarity for TFIDF')
        sim = pairwise_sim[0, :]
        for i in range(len(texts)):
            if sim[i] >= 0.25:
                print(texts[i], sim[i])
    
    if sim_tech=='word2vec':
        print('\nSimilarity for WORD2VEC')
        sim = pairwise_sim[0, :]
        for i in range(len(texts)):
            if sim[i] >= 0.905:
                print(texts[i], sim[i])

def show_least_avg_most_similar(texts, matrix_filename):
    texts        = study_ideas_dataset()
    pairwise_sim = np.load('storage/results/'+matrix_filename)
    sim_tech     = matrix_filename.split('_')[0]

    temp_max = 0.0
    temp_avg = 
    temp_min = 1.0
    for i in range(len(pairwise_sim)):
        for j in range(len(pairwise_sim)):
            if i == j: continue
            if pairwise_sim[i, j] > temp_max and pairwise_sim[i, j] < 0.999:
                temp_max = pairwise_sim[i, j]
                x_max_idx, y_max_idx = i, j
            if pairwise_sim[i, j] < temp_min:
                temp_min = pairwise_sim[i, j]
                x_min_idx, y_min_idx = i, j
            if np.abs(pairwise_sim[i, j] - temp_avg) < 0.1**6:
                x_avg_idx, y_avg_idx = i, j
    

    print('Very similar')
    print(x[x_max_idx])
    print(x[y_max_idx])
    print(pairwise_sim[x_max_idx, y_max_idx])

    print('Very dissimilar')
    print(x[x_min_idx])
    print(x[y_min_idx])
    print(pairwise_sim[x_min_idx, y_min_idx])

    print('Average similarity')
    print(x[x_avg_idx])
    print(x[y_avg_idx])
    print(pairwise_sim[x_avg_idx, y_avg_idx])
    


    

if __name__ == '__main__':

    texts           = study_ideas_dataset()
    matrix_filename = 'tfidf_similarity_blubb.npy'
    load_and_compare(texts, matrix_filename)
    quit()

    print('TFIDF EXAMPLES:')
    temp_max = 0.0
    temp_avg = 0.5
    temp_min = 1.0
    for i in range(len(pairwise_sim)):
        for j in range(len(pairwise_sim)):
            if i == j: continue
            if pairwise_sim[i, j] > temp_max and pairwise_sim[i, j] < 0.999:
                temp_max = pairwise_sim[i, j]
                x_max_idx, y_max_idx = i, j
            if pairwise_sim[i, j] < temp_min:
                temp_min = pairwise_sim[i, j]
                x_min_idx, y_min_idx = i, j
            if np.abs(pairwise_sim[i, j] - temp_avg) < 0.05:
                x_avg_idx, y_avg_idx = i, j

    print('Very similar')
    print(x[x_max_idx])
    print(x[y_max_idx])
    print(pairwise_sim[x_max_idx, y_max_idx])

    print('Very dissimilar')
    print(x[x_min_idx])
    print(x[y_min_idx])
    print(pairwise_sim[x_min_idx, y_min_idx])

    print('Average similarity')
    print(x[x_avg_idx])
    print(x[y_avg_idx])
    print(pairwise_sim[x_avg_idx, y_avg_idx])

    pairwise_sim = np.load('word2vec_similarity.npy')

    sim_1 = pairwise_sim[0, :]
    for i in range(len(x)):
        if sim_1[i] >= 0.85:
            print(x[i], sim_1[i])
    quit()

    print('WORD TO VEC EXAMPLES:')
    temp_max = 0.0
    temp_avg = 0.5
    temp_min = 1.0
    for i in range(len(pairwise_sim)):
        for j in range(len(pairwise_sim)):
            if i == j: continue
            if pairwise_sim[i, j] > temp_max and pairwise_sim[i, j] < 0.999:
                temp_max = pairwise_sim[i, j]
                x_max_idx, y_max_idx = i, j
            if pairwise_sim[i, j] < temp_min:
                temp_min = pairwise_sim[i, j]
                x_min_idx, y_min_idx = i, j
            if np.abs(pairwise_sim[i, j] - temp_avg) < 0.05:
                x_avg_idx, y_avg_idx = i, j

    print('Very similar')
    print(x[x_max_idx])
    print(x[y_max_idx])
    print(pairwise_sim[x_max_idx, y_max_idx])

    print('Very dissimilar')
    print(x[x_min_idx])
    print(x[y_min_idx])
    print(pairwise_sim[x_min_idx, y_min_idx])

    print('Average similarity')
    print(x[x_avg_idx])
    print(x[y_avg_idx])
    print(pairwise_sim[x_avg_idx, y_avg_idx])
