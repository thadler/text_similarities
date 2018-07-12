import numpy as np
import scipy

from model import *

import gensim.models.keyedvectors as word2vec

from data_reader import quora_duplicate_questions_dataset, batch


if __name__=='__main__':


    # hyperparameters
    batchsize    = 128
    lr           = 0.01
    display_step = 10

    # data
    x1_train, x2_train, x1_test, x2_test, y_train, y_test = quora_duplicate_questions_dataset()

    x1_train = [[w.strip() for w in text.split()] for text in x1_train]
    x2_train = [[w.strip() for w in text.split()] for text in x2_train]
    x1_test  = [[w.strip() for w in text.split()] for text in x1_test ]
    x2_test  = [[w.strip() for w in text.split()] for text in x2_test ]
    print('loading embedding')
    emb = word2vec.KeyedVectors.load_word2vec_format('~/Desktop/GoogleNews-vectors-negative300.bin', binary=True)
    print('done loading embedding')
    wv = emb.wv
    x1_train = [[w for w in text if w in wv] for text in x1_train]
    x2_train = [[w for w in text if w in wv] for text in x2_train]
    x1_test  = [[w for w in text if w in wv] for text in x1_test ]
    x2_test  = [[w for w in text if w in wv] for text in x2_test ]

    # model paramters
    x1, x2, y, k = placeholders()
    proj_op      = project_operation(x1,x2,k)

    infer_op = inference_operation(proj_op)
    pred_op  = predict_operation(infer_op)
    loss_op  = loss_operation(infer_op, y)
    acc_op   = accuracy_operation(pred_op, y)
    opti_op  = optimizer_operation(loss_op, lr)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for step in range(10**9):
            # train the neural network
            x1_tr, x2_tr, y_tr = batch(x1_train, x2_train, y_train, batchsize, emb)
            _, loss, pred, acc = sess.run([opti_op, loss_op, pred_op, acc_op], feed_dict={x1: x1_tr, x2: x2_tr, y: y_tr, k: 0.6})
            
            # eval NN
            if step%display_step==0:
                # steps and losses
                print('training losses for step: ', step)
                print('\tloss: ',                   loss)
                print('\ttraining accuracy: ',      acc)

                
                x1_te, x2_te, y_te = batch(x1_test, x2_test, y_test, 10*batchsize, emb)
                pred, acc = sess.run([pred_op, acc_op], feed_dict={x1: x1_te, x2: x2_te, y: y_te, k: 1.0})
                print('testing for step: ',   step)
                print('\ttesting accuracy: ', acc)







