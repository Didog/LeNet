# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 10:52:29 2019

@author: Lhy
"""

from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf

#Load Data
mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test = mnist.test.images, mnist.test.labels

#assert(len(X_train) == len(y_train))
#assert(len(X_validation) == len(y_validation))
#assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:  {} samples".format(len(X_train)))
print("Vaidation Set: {} samples".format(len(X_validation)))
print("Test Set:      {} samples".format(len(X_test)))



#28*28*1 --->  32*32*C
#pad images
X_train = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), "constant")
X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), "constant")
X_test = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), "constant")

print("Updated Image Shape: {}".format(X_train[0].shape))

#import random
#import matplotlib.pyplot as plt

#index = random.randint(0, len(X_train))
#image = X_train[index].squeeze()

#plt.figure(figsize=(1,1))
#plt.imshow(image, cmap='gray')
#print(y_train[index])


#X_train, y_train = shuffle(X_train, y_train)


EPOCHS = 10
BATCH_SIZE = 128



with tf.variable_scope("input"):
    x = tf.placeholder(tf.float32, [None, 32,32,1])
    y_ = tf.placeholder(tf.int32, [None])
    one_hot_y = tf.one_hot(y_, 10)

with tf.variable_scope("layer1-conv1"):
    #32*32*1---->28*28*6
    conv1_weight = tf.Variable(tf.truncated_normal(shape=[5,5,1,6], mean=0, stddev=0.1))
    conv1_bias = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_weight, strides=[1,1,1,1], padding='VALID')
    conv1_output = tf.nn.relu(conv1 + conv1_bias)
    
with tf.variable_scope("layer2-pooling1"):
    #28*28*6---->14*14*6
    pooling1_output = tf.nn.max_pool(conv1_output, ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID')

with tf.variable_scope("layer3-conv2"):
    #14*14*6 ---->10*10*16
    conv2_weight = tf.Variable(tf.truncated_normal(shape=[5,5,6, 16], mean=0, stddev=0.1))
    conv2_bias = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pooling1_output, conv2_weight, strides=[1,1,1,1], padding="VALID")
    conv2_output = tf.nn.relu(conv2 + conv2_bias)
    
with tf.variable_scope("layer4-pooling2"):
    #10*10*16 -----> 5*5*16
    pooling2_output = tf.nn.max_pool(conv2_output, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")
    
with tf.variable_scope("layer5-fc1"):
    #5*5*16 ------>1*1*120
    from tensorflow.contrib.layers import flatten
    fc0 = flatten(pooling2_output)   #5*5*16----->400
    
    fc1_weight = tf.Variable(tf.truncated_normal(shape=[400,120], mean=0, stddev=0.1))
    fc1_bias = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_weight)
    fc1_output = tf.nn.relu(fc1 + fc1_bias)
    
with tf.variable_scope("layer6-fc2"):
    #120 -----> 84
    fc2_weight = tf.Variable(tf.truncated_normal(shape=[120,84], mean=0, stddev=0.1))
    fc2_bias = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1_output, fc2_weight)
    fc2_output = tf.nn.relu(fc2 + fc2_bias)
    
with tf.variable_scope("layer7-output"):
    #84----->10
    fc3_weight = tf.Variable(tf.truncated_normal(shape=[84,10], mean=0, stddev=0.1))
    fc3_bias = tf.Variable(tf.zeros(10))
    fc3 = tf.matmul(fc2_output, fc3_weight) + fc3_bias
    
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=fc3)
loss = tf.reduce_mean(cross_entropy)
training_step = tf.train.AdamOptimizer(0.001).minimize(loss)

correction_prediction = tf.equal(tf.arg_max(fc3, 1), tf.argmax(one_hot_y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_op, feed_dict={x: batch_x, y_: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    num_example = len(X_train)
    print("Traing...")
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_example, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_step, feed_dict={x:batch_x, y_:batch_y})
        
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    
    