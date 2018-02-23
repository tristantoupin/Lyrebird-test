import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm_notebook
from numpy import random

epochs = 100
learning_rate = 0.001
batch_size = 100
hidden_layer_size = 256
number_of_layers = 1
dropout_rate = 0.5
number_of_classes = 3
gradient_clip_margin = 4
window_size = 57

model, session, restore_model, saver = None, None, None, None

def slipt_data_part_1(data, rd_seed, train_ratio, max_train_size = float("inf")):
    random.seed(seed = rd_seed)
    train_length = min(max_train_size, int(train_ratio * len(data)))
    print("Preparing train set...")
    X_train, y_train = window_data(data[:train_length], window_size)
    print("Preparing test set...")
    X_test, y_test = window_data(data[train_length:], window_size)

    return X_train, X_test, y_train, y_test


def window_data(data, window_size):
    X, y = [], []
    
    for s in tqdm_notebook(data):
        i = 0
        while (i + window_size) <= len(s) - 1:
            X.append(s[i:i+window_size])
            y.append(s[i+window_size])
            i += 1

    assert len(X) ==  len(y)
    return np.array(X), np.array(y)


def LSTM_cell(hidden_layer_size, batch_size, num_layers, dropout_rate = 0.5):
    layer = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size)
    layer = tf.contrib.rnn.DropoutWrapper(layer, output_keep_prob = dropout_rate)                              
    cell = tf.contrib.rnn.MultiRNNCell([layer] * num_layers)
    init_state = cell.zero_state(batch_size, tf.float32)
    return cell, init_state
                      
                                          
def output_layer(lstm_output, in_size, out_size):
    data = lstm_output[:, -1, :]
    weights = tf.Variable(tf.truncated_normal([in_size, out_size], stddev = 0.05), name = 'output_layer_weights')
    bias = tf.Variable(tf.zeros([out_size]), name = 'output_layer_bias')
    output = tf.matmul(data, weights) + bias
    return output
                                          

def optimize_loss(logits, targets, learning_rate, grad_clip_margin):
    losses = []
    for i in range(targets.get_shape()[0]):
        losses.append([(tf.pow(logits[i] - targets[i], 2))])
        
    loss = tf.reduce_sum(losses)/(2*batch_size)
    # Cliping gradient loss
    gradients = tf.gradients(loss, tf.trainable_variables())
    clipper_, _ = tf.clip_by_global_norm(gradients, grad_clip_margin)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_optimizer = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
    return loss, train_optimizer

                                          
class recurrent_neural_net(object):
    def __init__(self):
        self.inputs = tf.placeholder(tf.float32, [batch_size, window_size, number_of_classes], name='input_data')
        self.targets = tf.placeholder(tf.float32, [batch_size, number_of_classes], name='targets')
        cell, init_state = LSTM_cell(hidden_layer_size, batch_size, number_of_layers, dropout_rate)
        outputs, states = tf.nn.dynamic_rnn(cell, self.inputs, initial_state=init_state)
        self.logits = output_layer(outputs, hidden_layer_size, number_of_classes)
        self.loss, self.opt = optimize_loss(self.logits, self.targets, learning_rate, gradient_clip_margin)
    
    
def prep_model(restore = None):
    global model
    global session
    global restore_model
    global saver
    restore_model = restore
    tf.reset_default_graph()
    model = recurrent_neural_net()
    session =  tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    if (restore_model):
        saver.restore(session, "models/" + restore_model + ".ckpt")
        print("Model restored.")

        
def train(X_train, y_train, name="foo"):

    for i in tqdm_notebook(range(epochs)):
        traind_scores = []
        ii = 0
        epoch_loss = []
        pbar = tqdm_notebook(total = len(X_train)+1, leave=False)
        while(ii + batch_size) <= len(X_train):
            X_batch = X_train[ii:ii+batch_size]
            y_batch = y_train[ii:ii+batch_size]
            o, c, _ = session.run([model.logits, model.loss, model.opt], feed_dict={model.inputs:X_batch, model.targets:y_batch})
            epoch_loss.append(c)
            traind_scores.append(o)
            ii += batch_size
            pbar.update(batch_size)
        pbar.close()
        if (i % 10) == 0:
            print('Epoch {}/{}'.format(i, epochs), ' Current loss: {}'.format(np.mean(epoch_loss)))
        if (restore_model):
            saver.save(session, 'models/' + name + '.ckpt')


def test(input_): 
    print("Creating new stroke...")
    start = random.randint(0, high = len(input_) - 700)
    X_test = input_[start:start + 700]
    tests = []
    i = 0
    while i+batch_size <= 700:
        o = session.run([model.logits], feed_dict={model.inputs:X_test[i:i+batch_size]})
        i += batch_size
        np.append(X_test, o[0])
        tests.append(o)
        
    temp_test = (np.asarray(tests)).reshape(len(X_test),3)

    std_data = np.std(temp_test[:, 0])
    mean_data = np.mean(temp_test[:, 0])
    
    for count, item in enumerate(temp_test):
            if item[0] <= mean_data + std_data:
                temp_test[count][0] = int(0)
            else:
                temp_test[count][0] = int(1)
                
    return temp_test, start