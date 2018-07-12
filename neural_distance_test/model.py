import numpy as np
import tensorflow as tf

def placeholders():
    x1 = tf.placeholder("float", [None, 300*3]) # mean min max
    x2 = tf.placeholder("float", [None, 300*3]) # mean min max
    y  = tf.placeholder("float", [None, 2])
    k  = tf.placeholder("float", [])
    return x1, x2, y, k


def project_operation(x1, x2, k):
    #concat_1 = tf.concat([x1, x2, x1-x2, x1*x2], axis=1)
    concat_1 = tf.concat([x1, x2], axis=1)
    
    embed_1  = tf.contrib.layers.fully_connected(concat_1, 30, activation_fn=None)
    embed_11 = tf.contrib.layers.fully_connected(concat_1, 30, activation_fn=tf.nn.elu)

    concat_2 = tf.concat([embed_1, embed_11], axis=1)
    proj  = tf.contrib.layers.fully_connected(concat_2, 40, activation_fn=tf.nn.elu)
    drop  = tf.nn.dropout(proj, keep_prob=k)
    proj2 = tf.contrib.layers.fully_connected(concat_2, 20, activation_fn=tf.nn.elu)
    return proj2


def inference_operation(project_op):
    return tf.contrib.layers.fully_connected(project_op, 2, activation_fn=None)


def predict_operation(infer_op):
    return tf.argmax(infer_op, 1)


def accuracy_operation(pred_op, y):
    correct_prediction = tf.equal(pred_op, tf.argmax(y, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, "float"))


def loss_operation(infer_op, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=infer_op, labels=y))


def optimizer_operation(loss_op, learning_rate):
    return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)



if __name__=='__main__':
    lr           = 0.01
    x1, x2, y, k = placeholders()
    proj_op      = project_operation(x1,x2,k)

    infer_op = inference_operation(proj_op)
    pred_op  = predict_operation(infer_op)
    acc_op   = accuracy_operation(pred_op, y)
    loss_op  = loss_operation(infer_op, y)
    opti_op  = optimizer_operation(loss_op, lr)
