import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm_notebook

strokes = np.load('../data/strokes.npy')
stroke = strokes[0]

epochs = 100
learning_rate = 0.001
batch_size = 10
hidden_layer_size = 256
number_of_layers = 1
dropout_rate = 0.5
number_of_classes = 3
gradient_clip_margin = 4
window_size = 50

def window_data(data, window_size):
    X = []
    y = []
    
    i = 0
    while (i + window_size) <= len(data) - 1:
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
        i += 1

    assert len(X) ==  len(y)
    return X, y


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
    
    #Cliping the gradient loss
    gradients = tf.gradients(loss, tf.trainable_variables())
    clipper_, _ = tf.clip_by_global_norm(gradients, grad_clip_margin)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_optimizer = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
    return loss, train_optimizer

                                          
class recurrent_neural_net(object):
    def __init__(self):
    
        self.inputs = tf.placeholder(tf.float32, [batch_size, window_size, 3], name='input_data')
        self.targets = tf.placeholder(tf.float32, [batch_size, 3], name='targets')
        
        cell, init_state = LSTM_cell(hidden_layer_size, batch_size, number_of_layers, dropout_rate)
        outputs, states = tf.nn.dynamic_rnn(cell, self.inputs, initial_state=init_state)
        self.logits = output_layer(outputs, hidden_layer_size, number_of_classes)
        self.loss, self.opt = optimize_loss(self.logits, self.targets, learning_rate, gradient_clip_margin)
    
    
tf.reset_default_graph()
model = recurrent_neural_net()

session =  tf.Session()
session.run(tf.global_variables_initializer())


def train(X_train, y_train):

    for i in tqdm_notebook(range(epochs)):
        traind_scores = []
        ii = 0
        epoch_loss = []
        while(ii + batch_size) <= len(X_train):
            X_batch = X_train[ii:ii+batch_size]
            y_batch = y_train[ii:ii+batch_size]
            o, c, _ = session.run([model.logits, model.loss, model.opt], feed_dict={model.inputs:X_batch, model.targets:y_batch})

            epoch_loss.append(c)
            traind_scores.append(o)
            ii += batch_size
        if (i % 10) == 0:
            print('Epoch {}/{}'.format(i, epochs), ' Current loss: {}'.format(np.mean(epoch_loss)))

def test(X_test):       
    tests = []
    i = 0
    while i+batch_size <= len(X_test):
        o = session.run([model.logits], feed_dict={model.inputs:X_test[i:i+batch_size]})
        i += batch_size
        tests.append(o)
    return tests

def generate_unconditionally(random_seed=1):
    # Input:
    #   random_seed - integer
    X, y = window_data(strokes[0], window_size)

    X_train  = np.array(X[:100])
    y_train = np.array(y[:100])

    X_test = np.array(X[:100])
    y_test = np.array(y[:100])

    print("X_train size: {}".format(X_train.shape))
    print("y_train size: {}".format(y_train.shape))
    print("X_test size: {}".format(X_test.shape))
    print("y_test size: {}".format(y_test.shape))

    train(X_train, y_train)
    

    temp_test = np.asarray(test(X_test))
    temp_test = (temp_test).reshape(len(X_test),3)


    for count, item in enumerate(temp_test):
        if item[0] < 0.5:
            temp_test[count][0] = 0
        else:
            temp_test[count][0] = 1
    print(temp_test.shape)
    print(temp_test)
    
    
    # Output:
    #   stroke - numpy 2D-array (T x 3)
    return (temp_test)


def generate_conditionally(text='welcome to lyrebird', random_seed=1):
    # Input:
    #   text - str
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)
    return stroke


def recognize_stroke(stroke):
    # Input:
    #   stroke - numpy 2D-array (T x 3)

    # Output:
    #   text - str
    return 'welcome to lyrebird'