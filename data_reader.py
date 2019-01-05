import numpy as np
import csv
import re

def load_train_texts_and_similarities():
    with open('storage/datasets/sts-train.csv') as f:
        data = f.readlines()
        data = [data[i].split('\t') for i in range(len(data))]
        texts = [[data[i][5].replace(',','').replace('.','').strip(), data[i][6].replace('.','').replace(',','').strip()] for i in range(len(data))]
        sims  = np.array([float(data[i][4]) for i in range(len(data))])
    return texts, sims

def load_dev_texts_and_similarities():
    with open('storage/datasets/sts-dev.csv') as f:
        data = f.readlines()
        data = [data[i].split('\t') for i in range(len(data))]
        texts = [[data[i][5].replace(',','').replace('.','').strip(), data[i][6].replace('.','').replace(',','').strip()] for i in range(len(data))]
        sims  = np.array([float(data[i][4]) for i in range(len(data))])
    return texts, sims

def load_test_texts_and_similarities():
    with open('storage/datasets/sts-test.csv') as f:
        data = f.readlines()
        data = [data[i].split('\t') for i in range(len(data))]
        texts = [[data[i][5].replace(',','').replace('.','').strip(), data[i][6].replace('.','').replace(',','').strip()] for i in range(len(data))]
        sims  = np.array([float(data[i][4]) for i in range(len(data))])
    return texts, sims

def load_chi19ppre_partial_approved():
    with open('storage/datasets/chi19ppre-partial-approved.csv') as f:
        data = csv.reader(f, delimiter=',', quotechar='"')
        texts = [line[2].replace('\n', '') for line in data]
        sims = []
    return texts, sims

def load_chi19p_C1_complete():
    with open('storage/datasets/chi19p-C1-complete.csv') as f:
        data  = f.readlines()
        data  = [line.replace('"','').replace('\n','').replace(',',' ,').replace('.',' .') for line in data[1:]]
        data  = [line.split(',') for line in data]
        texts = [line[2] for line in data]
        sims  = []
    return texts, sims

def load_chi19p_C3_complete():
    with open('storage/datasets/chi19p-c3-complete.csv') as f:
        data  = f.readlines()
        data  = [line.replace('"','').replace('\n','').replace(',',' ,').replace('.',' .') for line in data[1:]]
        data  = [line.split(',') for line in data]
        texts = [line[2] for line in data]
        sims  = []
    return texts, sims



if __name__=='__main__':
    load_train_texts_and_similarities()
    load_dev_texts_and_similarities()
    load_test_texts_and_similarities()
    load_chi19ppre_partial_approved()
    load_chi19p_C1_complete()
    load_chi19p_C3_complete()
