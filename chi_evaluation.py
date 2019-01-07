import sys
from data_reader import *
from scipy.stats import pearsonr, spearmanr


def get_ai_txt_2_sims(ai_txts, ai_sims, ni_txt_2_sims):
    """
    The ai procedures calculate ALL similarities. Only some are manually 
    annotated however. So we must select the ai-similarities relevant for the
    comparison.
    """
    
    txt_2_idx = dict((t.replace('  ',' ').strip(), i) for i,t in enumerate(ai_txts))
    dictionary = dict()
    for i, k in enumerate(ni_txt_2_sims):
        t1, t2 = k[0].replace('  ',' ').strip(), k[1].replace('  ',' ').strip()
        idx1, idx2 = txt_2_idx[t1], txt_2_idx[t2]
        dictionary[(t1,t2)] = ai_sims[idx1,idx2]
        dictionary[(t2,t1)] = ai_sims[idx1,idx2]
    return dictionary


def load_data_chi(chi_nr, manual_fn, automatic_fn):
    ni_txt_2_sims = load_ni_sims(manual_fn)
    if chi_nr==1: ai_txts, ai_sims = load_chi19p_C1_complete(automatic_fn)
    if chi_nr==2: ai_txts, ai_sims = load_chi19p_C2_complete(automatic_fn)
    if chi_nr==3: ai_txts, ai_sims = load_chi19p_C3_complete(automatic_fn)
    ai_txt_2_sims = get_ai_txt_2_sims(ai_txts, ai_sims, ni_txt_2_sims)
    return ni_txt_2_sims, ai_txt_2_sims


def get_correlation_chi(ni_txt_2_sims, ai_txt_2_sims):
    n = len(ni_txt_2_sims.keys())
    correct_values   = np.zeros(n)
    predicted_values = np.zeros(n)
    for i, k in enumerate(sorted(ni_txt_2_sims.keys())):
        correct_values[i]   = ni_txt_2_sims[k]
        t1, t2 = k[0].replace('  ',' ').strip(), k[1].replace('  ',' ').strip()
        predicted_values[i] = ai_txt_2_sims[(t1,t2)]
    # pearson's correlation coefficient requires: linear relationship, normal distributed deviation 
    # spearman's correlation coefficient
    pc = pearsonr(correct_values, predicted_values)[0]
    sc = spearmanr(correct_values, predicted_values)[0]
    print("spearman's rank-order correlation coefficient:\n", sc)
    print("pearson's correlation coefficient:\n",             pc)
    return pc, sc

    
def check_input(chi_nr, manual_fn, automatic_fn):
    if 'ratings' not in manual_fn:
        print("Not the correct filename, doesn't contain 'ratings'")
        quit()
    if 'c'+str(chi_nr) not in manual_fn:
        print("Not the correct filename, doesn't contain correct chi_nr")
        quit()


if __name__ == '__main__':

    script       = sys.argv[0]
    chi_nr       = sys.argv[1] # which chi texts to load
    manual_fn    = sys.argv[2] # manually annotated similarities filename
    automatic_fn = sys.argv[3] # machine  predicted similarities filename 

    # assertions
    check_input(chi_nr, manual_fn, automatic_fn)

    # load the data
    ni_txt_2_sims, ai_txt_2_sims = load_data_chi(int(chi_nr), manual_fn, automatic_fn)

    # calculate peason and spearman correlation
    get_correlation_chi(ni_txt_2_sims, ai_txt_2_sims)
    
