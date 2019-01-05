import sys
from data_reader import *
from scipy.stats import pearsonr, spearmanr

if __name__ == '__main__':

    script = sys.argv[0]
    ai_sims  = sys.argv[1] # manually annotated similarities filename 
    ni_sims  = sys.argv[2] # machine  predicted similarities filename

    if 'ratings' not in ai_sims:
        print("Not the correct filename, doesn't contain 'ratings'")
        quit()

    # load the data
    ni_txt_2_sims = load_ni_sims(ni_sims)
    #ai_sims = np.array(list(map(float, open(ai_sims).readlines())))
    
    # pearson's correlation coefficient requires:
    # linear relationship
    # normal distributed deviation 
    print("pearson's correlation coefficient:\n", pearsonr(correct_values, predicted_values)[0])
    
    # spearman's correlation coefficient
    print("spearman's rank-order correlation coefficient:\n", spearmanr(correct_values, predicted_values)[0])
