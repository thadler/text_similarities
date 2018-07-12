import csv
import numpy as np
import scipy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier


def quora_duplicate_questions_dataset():
    with open("datasets/quora_duplicate_questions.tsv") as fin:
        filereader = csv.reader(fin, delimiter='\t')
        x11, x21, y1 = [], [], []
        first = True
        for _, _, _, q1, q2, dup in filereader:
            if first: first = False; continue
            x11.append(q1)
            x21.append(q2)
            y1.append(int(dup))

    # balance the dataset
    n_pos = np.sum(y1)
    c_neg = 0
    x1, x2, y = [], [], []
    for i in range(len(x11)):
        if c_neg == n_pos and y1[i] == 0: continue
        x1.append(x11[i])
        x2.append(x21[i])
        y.append(y1[i])
        if y1[i] == 0: c_neg += 1

    x_train, x_test, y_train, y_test = train_test_split(list(zip(x1, x2)), y, test_size=0.1, random_state=42)
    x1_train = [x[0] for x in x_train]
    x2_train = [x[1] for x in x_train]
    x1_test = [x[0] for x in x_test]
    x2_test = [x[1] for x in x_test]
    return x1_train, x2_train, x1_test, x2_test, y_train, y_test


def study_ideas_dataset():
    x = []
    with open("storage/datasets/preliminary-study-ideas.csv") as fin:
        filereader = csv.reader(fin, delimiter='\t')
        for _, txt in filereader:
            x.append(txt.strip().replace('\n', ''))
    return x
