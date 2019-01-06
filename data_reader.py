import numpy as np
import csv
import re

def load_train_texts_and_similarities():
    with open('storage/datasets/sts-train.csv') as f:
        data  = f.readlines()
        data  = [data[i].split('\t') for i in range(len(data))]
        texts = [[data[i][5].replace(',','').replace('.','').strip(), data[i][6].replace('.','').replace(',','').strip()] for i in range(len(data))]
        sims  = np.array([float(data[i][4]) for i in range(len(data))])
    return texts, sims

def load_dev_texts_and_similarities():
    with open('storage/datasets/sts-dev.csv') as f:
        data  = f.readlines()
        data  = [data[i].split('\t') for i in range(len(data))]
        texts = [[data[i][5].replace(',','').replace('.','').strip(), data[i][6].replace('.','').replace(',','').strip()] for i in range(len(data))]
        sims  = np.array([float(data[i][4]) for i in range(len(data))])
    return texts, sims

def load_test_texts_and_similarities():
    with open('storage/datasets/sts-test.csv') as f:
        data  = f.readlines()
        data  = [data[i].split('\t') for i in range(len(data))]
        texts = [[data[i][5].replace(',','').replace('.','').strip(), data[i][6].replace('.','').replace(',','').strip()] for i in range(len(data))]
        sims  = np.array([float(data[i][4]) for i in range(len(data))])
    return texts, sims


def load_chi19ppre_partial_approved(sim_prediction_filename=None):
    with open('storage/datasets/chi19ppre-partial-approved.csv') as f:
        data  = csv.reader(f, delimiter=',', quotechar='"')
        texts = [line[2].replace('\n', '') for line in data]
    sims  = []
    if sim_prediction_filename is not None:    
        sims = np.loadtxt(sim_prediction_filename)
    return texts, sims

def load_chi19p_C1_complete(sim_prediction_filename=None):
    with open('storage/datasets/chi19p-C1-complete.csv') as f:
        data  = csv.reader(f, delimiter=',', quotechar='"')
        data  = [line[2] for line in data]
        texts = [line.replace('"','').replace('\n','').replace(',',' ,').replace('.',' .') for line in data[1:]]
    sims  = []
    if sim_prediction_filename is not None:    
        sims = np.loadtxt(sim_prediction_filename)
    return texts, sims

def load_chi19p_C2_complete(sim_prediction_filename=None):
    with open('storage/datasets/chi19p-C3-complete.csv') as f:
        data  = csv.reader(f, delimiter=',', quotechar='"')
        data  = [line[2] for line in data]
        texts = [line.replace('"','').replace('\n','').replace(',',' ,').replace('.',' .') for line in data[1:]]
    sims  = []
    if sim_prediction_filename is not None:    
        sims = np.loadtxt(sim_prediction_filename)
    return texts, sims

def load_chi19p_C3_complete(sim_prediction_filename=None):
    with open('storage/datasets/chi19p-C3-complete.csv') as f:
        data  = csv.reader(f, delimiter=',', quotechar='"')
        data  = [line[2] for line in data]
        texts = [line.replace('"','').replace('\n','').replace(',',' ,').replace('.',' .') for line in data[1:]]
    sims  = []
    if sim_prediction_filename is not None:    
        sims = np.loadtxt(sim_prediction_filename)
    return texts, sims

def load_ni_sims(fname): # ni = natural intelligence
    """
    returns a dict object with (t1,t2) -> similarity <- (t2,t1)
    """
    with open(fname) as f:
        data = csv.reader(f, delimiter=',', quotechar='"')
        data = [line for line in data]
        preprocess = lambda txt: txt.replace('"','').replace('\n','').replace(',',' ,').replace('.',' .')
        dictionary = dict()
        for l in data:
            dictionary[(preprocess(l[0]), preprocess(l[1]))] = float(l[2])
            dictionary[(preprocess(l[1]), preprocess(l[0]))] = float(l[2])
    return dictionary

if __name__=='__main__':
    load_train_texts_and_similarities()
    load_dev_texts_and_similarities()
    load_test_texts_and_similarities()
    load_chi19ppre_partial_approved()
    load_chi19p_C1_complete()
    load_chi19p_C3_complete()
    load_ni_sims('storage/datasets/chi19p-c1-ratings.csv')
    load_chi19p_C3_complete('storage/predictions/chi19p-C1-complete/avg_wv_fasttext_sim_cosine.txt')
