# imports
import numpy as np
import matplotlib.pyplot as plt
import gensim.models.keyedvectors as word2vec
from data_reader import study_ideas_dataset


def load_and_show_similar_to_first_example(texts, matrix_filename):
    texts        = study_ideas_dataset()
    pairwise_sim = np.loadtxt('storage/results/'+matrix_filename)
    sim_tech     = matrix_filename.split('_')[0]

    if sim_tech=='lsa':
        print('\nSimilarity for LSA')
        sim = pairwise_sim[0, :]
        for i in range(len(texts)):
            if sim[i] >= 0.28:
                print(texts[i], sim[i])
    elif sim_tech=='tfidf':
        print('\nSimilarity for TFIDF')
        sim = pairwise_sim[0, :]
        for i in range(len(texts)):
            if sim[i] >= 0.25:
                print(texts[i], sim[i])
    elif sim_tech=='word2vec':
        print('\nSimilarity for WORD2VEC')
        sim = pairwise_sim[0, :]
        for i in range(len(texts)):
            if sim[i] >= 0.905:
                print(texts[i], sim[i])
    else:
        print('similarity technique is not offered')

def show_least_avg_most_similar(texts, matrix_filename):
    texts        = study_ideas_dataset()
    pairwise_sim = np.loadtxt('storage/results/'+matrix_filename)
    sim_tech     = matrix_filename.split('_')[0]

    temp_max = 0.0
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
    flat_idx = np.abs(pairwise_sim - np.mean(pairwise_sim)).argmin()
    x_avg_idx, y_avg_idx = flat_idx//pairwise_sim.shape[0], flat_idx%pairwise_sim.shape[1]

    print('Very dissimilar')
    print(texts[x_min_idx])
    print(texts[y_min_idx])
    print(pairwise_sim[x_min_idx, y_min_idx])

    print('Average similarity')
    print(texts[x_avg_idx])
    print(texts[y_avg_idx])
    print(pairwise_sim[x_avg_idx, y_avg_idx])

    print('Very similar')
    print(texts[x_max_idx])
    print(texts[y_max_idx])
    print(pairwise_sim[x_max_idx, y_max_idx])


def distance_visualization(matrix_filename, bins=10):
    pairwise_sim = np.loadtxt('storage/results/'+matrix_filename)
    upper_idxs   = np.triu_indices(pairwise_sim.shape[0])
    dists        = pairwise_sim[upper_idxs]
    n, bins, patches = plt.hist(dists, bins, facecolor='blue', alpha=0.75)
    plt.xlabel('idea distance')
    plt.ylabel('idea count')
    plt.grid(True)
    plt.title('Histogram of Idea Distances')
    plt.show()
    

if __name__ == '__main__':

    texts           = study_ideas_dataset()
    matrix_filename = 'word2vec_similarity_blubb.csv'
    #load_and_show_similar_to_first_example(texts, matrix_filename)
    #show_least_avg_most_similar(texts, matrix_filename)
    distance_visualization(matrix_filename, bins=20)



