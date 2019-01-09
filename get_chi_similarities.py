import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from chi_evaluation import *

def get_all_sims(chi_nr, path_to_artificial, fname_manual):
    techniques = [f for f in os.listdir(path_to_artificial)]
    pcorrelation = []
    scorrelation = []
    for f in techniques:
        check_input(chi_nr, fname_manual, path_to_artificial+f)
        ni_txt_2_sims, ai_txt_2_sims = load_data_chi(int(chi_nr), fname_manual, path_to_artificial+f)
        pc, sc = get_correlation_chi(ni_txt_2_sims, ai_txt_2_sims, print_it=False)
        pcorrelation.append(pc)
        scorrelation.append(sc)
    techniques = [f[:-4] for f in techniques]
    table = pd.DataFrame(index=techniques, data={"pearson's correlation coefficient": pcorrelation})#, "spearman's correlation coefficient": scorrelation})
    print(table)

def presentable():
    techniques = [f for f in os.listdir('storage/predictions/chi19p-C1-complete/')]
    pcorrelation1 = []
    pcorrelation2 = []
    pcorrelation3 = []
    pcorrelation0 = []
    
    chi_nr = 1; 
    fname_manual='storage/datasets/chi19p-c1-ratings.csv'
    path_to_artificial = 'storage/predictions/chi19p-C1-complete/'
    for f in techniques:
        check_input(chi_nr, fname_manual, path_to_artificial+f)
        ni_txt_2_sims, ai_txt_2_sims = load_data_chi(int(chi_nr), fname_manual, path_to_artificial+f)
        pc, _ = get_correlation_chi(ni_txt_2_sims, ai_txt_2_sims, print_it=False)
        pcorrelation1.append(pc)
    chi_nr = 2; 
    fname_manual='storage/datasets/chi19p-c2-ratings.csv'
    path_to_artificial = 'storage/predictions/chi19p-C2-complete/'
    for f in techniques:
        check_input(chi_nr, fname_manual, path_to_artificial+f)
        ni_txt_2_sims, ai_txt_2_sims = load_data_chi(int(chi_nr), fname_manual, path_to_artificial+f)
        pc, _ = get_correlation_chi(ni_txt_2_sims, ai_txt_2_sims, print_it=False)
        pcorrelation2.append(pc)
    chi_nr = 3; 
    fname_manual='storage/datasets/chi19p-c3-ratings.csv'
    path_to_artificial = 'storage/predictions/chi19p-C3-complete/'
    for f in techniques:
        check_input(chi_nr, fname_manual, path_to_artificial+f)
        ni_txt_2_sims, ai_txt_2_sims = load_data_chi(int(chi_nr), fname_manual, path_to_artificial+f)
        pc, _ = get_correlation_chi(ni_txt_2_sims, ai_txt_2_sims, print_it=False)
        pcorrelation3.append(pc)
    
    chi_nr = 0;
    fname_manual='storage/datasets/chi19p-c2-ratings.csv'
    path_to_artificial = 'storage/predictions/chi19p-C2-complete_replace_hit/'
    for f in techniques:
        #check_input(chi_nr, fname_manual, path_to_artificial+f)
        ni_txt_2_sims, ai_txt_2_sims = load_data_chi(int(chi_nr), fname_manual, path_to_artificial+f)
        pc, _ = get_correlation_chi(ni_txt_2_sims, ai_txt_2_sims, print_it=False)
        pcorrelation0.append(pc)
    
    techniques = [f[:-4] for f in techniques]
    table = pd.DataFrame(index=techniques, data={"C1": pcorrelation1, "C2": pcorrelation2, "C3": pcorrelation3, "C2_2": pcorrelation0})#, "spearman's correlation coefficient": scorrelation})
    print(table)
    

if __name__=='__main__':
    """
    print('chi c1')
    get_all_sims(1, 'storage/predictions/chi19p-C1-complete/', 'storage/datasets/chi19p-c1-ratings.csv')
    print('chi c2')
    get_all_sims(2, 'storage/predictions/chi19p-C2-complete/', 'storage/datasets/chi19p-c2-ratings.csv')
    print('chi c3')
    get_all_sims(3, 'storage/predictions/chi19p-C3-complete/', 'storage/datasets/chi19p-c3-ratings.csv')
    """
    presentable()
