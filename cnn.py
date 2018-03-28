# simple python script to train a 1-layer neural network to classify cifar10 images use the tensorflow library
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os
# class written to replicate input_data from tensorflow.examples.tutorials.mnist for CIFAR-10
import cifar10_read


# location of the CIFAR-10 dataset
#CHANGE THIS PATH TO THE LOCATION OF THE CIFAR-10 dataset on your local machine
data_dir = '/home/wasp/assignment3/Datasets/'

# read in the dataset
print('reading in the CIFAR10 dataset')
dataset = cifar10_read.read_data_sets(data_dir, one_hot=True, reshape=False)   

using_tensorboard = True


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=.01))
    biases = tf.Variable(tf.constant(0.1, shape=[out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def weight_variable(shape): 
	#initalv = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

def bias_variable(shape): 
	#initialv = tf.constant(0.1,shape=shape) 
	return tf.Variable(tf.constant(0.1,shape=shape))

#stride 1 x_movement, y_movement, 1
def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME') 

def max_pool_3x3(x): 
	return tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')




##################################################
# PHASE 1  - ASSEMBLE THE GRAPH

# 1.1) define the placeholders for the input data and the ground truth labels

# x_input can handle an arbitrary number of input vectors of length input_dim = d 
# y_  are the labels (each label is a length 10 one-hot encoding) of the inputs in x_input
# If x_input has shape [N, input_dim] then y_ will have shape [N, 10]

input_dim = 32*32*3    # d
#x_input = tf.placeholder(tf.float32, shape = [None, input_dim])
x_input = tf.placeholder(tf.float32, shape = [None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, shape = [None, 10])


# 1.2) define the parameters of the network
# W: 3072 x 10 weight matrix,  b: bias vector of length 10



#W = tf.Variable(tf.truncated_normal([input_dim, 10], stddev=.01))
#b = tf.Variable(tf.constant(0.1, shape=[10]))

# 1.3) define the sequence of operations in the network to produce the output
# y = W *  x_input + b 
# y will have size [N, 10]  if x_input has size [N, input_dim]
#y = tf.matmul(x_input, W) + b

#hiddenlayer = add_layer(x_input, input_dim, 100, activation_function=tf.nn.relu)
#y = add_layer(hiddenlayer, 100, 10, activation_function=None)


W_conv1=weight_variable([5,5,3,64])
b_conv1=bias_variable([64])
#X1
h_conv1=tf.nn.relu(conv2d(x_input,W_conv1)+b_conv1) #32,32,64
h_pool1=max_pool_3x3(h_conv1) #16,16,64 
#conv layer2
W_conv2=weight_variable([5,5,64,128])
b_conv2=bias_variable([128])
#X2
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_3x3(h_conv2) #8,8,64

h_pool2_flat=tf.reshape(h_pool2,[-1,8*8*128]) 
W_fc1=weight_variable([8*8*128,1024]) 
b_fc1=bias_variable([1024])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
#keep_drop =tf.placeholder(tf.float32)
#h_fc1_drop=tf.nn.dropout(h_fc1,keep_drop)
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
y=tf.matmul(h_fc1,W_fc2)+b_fc2

# 1.4) define the loss funtion 
# cross entropy loss: 
# Apply softmax to each output vector in y to give probabilities for each class then compare to the ground truth labels via the cross-entropy loss and then compute the average loss over all the input examples
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# 1.5) Define the optimizer used when training the network ie gradient descent or some variation.
# Use gradient descent with a learning rate of .01
learning_rate = .01
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# (optional) definiton of performance measures
# definition of accuracy, count the number of correct predictions where the predictions are made by choosing the class with highest score
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  

# 1.6) Add an op to initialize the variables.
init = tf.global_variables_initializer()

##################################################


# If using TENSORBOARD
if using_tensorboard:
    # keep track of the loss and accuracy for the training set
    tf.summary.scalar('training loss', cross_entropy, collections=['training'])
    tf.summary.scalar('training accuracy', accuracy, collections=['training'])
    # merge the two quantities
    tsummary = tf.summary.merge_all('training')
    
    # keep track of the loss and accuracy for the validation set
    tf.summary.scalar('validation loss', cross_entropy, collections=['validation'])
    tf.summary.scalar('validation accuracy', accuracy, collections=['validation'])
    # merge the two quantities
    vsummary = tf.summary.merge_all('validation')

##################################################


##################################################
# PHASE 2  - PERFORM COMPUTATIONS ON THE GRAPH

n_iter = 1000

# 2.1) start a tensorflow session
with tf.Session() as sess:

    ##################################################
    # If using TENSORBOARD
    if using_tensorboard:
        # set up a file writer and directory to where it should write info + 
        # attach the assembled graph
        summary_writer = tf.summary.FileWriter('network1/results/exe2.3', sess.graph)
    ##################################################

    # 2.2)  Initialize the network's parameter variables
    # Run the "init" op (do this when training from a random initialization)
    sess.run(init) 

    # 2.3) loop for the mini-batch training of the network's parameters
    for i in range(n_iter):
        
        # grab a random batch (size nbatch) of labelled training examples
        nbatch = 200
        batch = dataset.train.next_batch(nbatch)

        # create a dictionary with the batch data 
        # batch data will be fed to the placeholders for inputs "x_input" and labels "y_"
        batch_dict = {
            x_input: batch[0], # input data
            y_: batch[1], # corresponding labels
         }
        
        # run an update step of mini-batch by calling the "train_step" op 
        # with the mini-batch data. The network's parameters will be updated after applying this operation
        sess.run(train_step, feed_dict=batch_dict)

        # periodically evaluate how well training is going
        if i % 50 == 0:

            # compute the performance measures on the training set by
            # calling the "cross_entropy" loss and "accuracy" ops with the training data fed to the placeholders "x_input" and "y_"
            
            tr = sess.run([cross_entropy, accuracy], feed_dict = {x_input:dataset.train.images[1:5000], y_: dataset.train.labels[1:5000]})

            # compute the performance measures on the validation set by
            # calling the "cross_entropy" loss and "accuracy" ops with the validation data fed to the placeholders "x_input" and "y_"

            val = sess.run([cross_entropy, accuracy], feed_dict={x_input:dataset.validation.images, y_:dataset.validation.labels})            

            info = [i] + tr + val
            print(info)

            ##################################################
            # If using TENSORBOARD
            if using_tensorboard:

                # compute the summary statistics and write to file
                summary_str = sess.run(tsummary, feed_dict = {x_input:dataset.train.images, y_: dataset.train.labels})
                summary_writer.add_summary(summary_str, i)

                summary_str1 = sess.run(vsummary, feed_dict = {x_input:dataset.validation.images, y_: dataset.validation.labels})
                summary_writer.add_summary(summary_str1, i)
            ##################################################

    # evaluate the accuracy of the final model on the test data
    test_acc = sess.run(accuracy, feed_dict={x_input: dataset.test.images, y_: dataset.test.labels})
    final_msg = 'test accuracy:' + str(test_acc)
    print(final_msg)

##################################################
