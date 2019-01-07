import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from chi_evaluation import *

def get_all_sims(chi_nr, path_to_artificial, fname_manual):
    techniques = [f for f in os.listdir(path_to_artificial)]
    techniques.remove('binary_sim.txt')
    pcorrelation = []
    scorrelation = []
    for f in techniques:
        check_input(chi_nr, fname_manual, path_to_artificial+f)
        ni_txt_2_sims, ai_txt_2_sims = load_data_chi(int(chi_nr), fname_manual, path_to_artificial+f)
        pc, sc = get_correlation_chi(ni_txt_2_sims, ai_txt_2_sims)
        pcorrelation.append(pc)
        scorrelation.append(sc)
    techniques = [f[:-4] for f in techniques]
    table = pd.DataFrame({"techniques": techniques, "pearson's correlation coefficient": pcorrelation, "spearman's correlation coefficient": scorrelation})
    print(table)
    
get_all_sims(1, 'storage/predictions/chi19p-C1-complete/', 'storage/datasets/chi19p-c1-ratings.csv')
get_all_sims(2, 'storage/predictions/chi19p-C2-complete/', 'storage/datasets/chi19p-c2-ratings.csv')
get_all_sims(3, 'storage/predictions/chi19p-C3-complete/', 'storage/datasets/chi19p-c3-ratings.csv')

