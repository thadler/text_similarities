import sys
from data_reader import *
from scipy.stats import pearsonr, spearmanr

if __name__ == '__main__':

    script = sys.argv[0]
    sims   = sys.argv[1] # train, dev or test 
    fname  = sys.argv[2] # predicted values filename

    if sims.startswith('sts'):
        print('ONLY enter train dev or test as string, NOT the filename')
        quit()

    # load the data
    _, correct_values = load_train_texts_and_similarities() if sims=='train' else load_dev_texts_and_similarities() if sims=='dev' else load_test_texts_and_similarities()
    predicted_values  = np.array(list(map(float, open(fname).readlines())))
    
    # pearson's correlation coefficient requires:
    # linear relationship
    # normal distributed deviation 
    print("pearson's correlation coefficient:\n", pearsonr(correct_values, predicted_values)[0])
    
    # spearman's correlation coefficient
    print("spearman's rank-order correlation coefficient:\n", spearmanr(correct_values, predicted_values)[0])
    
