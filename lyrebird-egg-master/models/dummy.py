import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm_notebook

strokes = np.load('../data/strokes.npy')
stroke = strokes[0]

epochs = 10
learning_rate = 0.01
batch_size = 100
hidden_layer_size = 700
number_of_layers = 1
dropout_rate = 0.5
number_of_classes = 3
gradient_clip_margin = 4
window_size = 57

def prep_data(data):
    X = []
    y = []
    all_data = []
    
    for quote in data:
        for pos in quote:
            all_data.append(pos)
    return all_data


def get_windows(data, start):
    i = 0
    X = []
    y = []
    while (i + window_size) <= (batch_size + window_size) - 1:
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
saver = tf.train.Saver()

session =  tf.Session()
session.run(tf.global_variables_initializer())


def train(all_data):
    for i in tqdm_notebook(range(epochs)):
        traind_scores = []
        ii = 0
        epoch_loss = []
        while(ii + batch_size) <= len(all_data):

            X_batch, y_batch = get_windows(all_data, ii)
            o, c, _ = session.run([model.logits, model.loss, model.opt], feed_dict={model.inputs:X_batch, model.targets:y_batch})
            epoch_loss.append(c)
            traind_scores.append(o)
            ii += batch_size
        if (i % 10) == 0:
            print('Epoch {}/{}'.format(i, epochs), ' Current loss: {}'.format(np.mean(epoch_loss)))

def test(X_test):       

    tests = []
    print(len(X_test))
    print((X_test))
    i = 0
    output_length = 700
    while i + batch_size <= output_length:
        o = session.run([model.logits], feed_dict={model.inputs:X_test})
        i += batch_size
        tests.append(o)
        
    print(len(tests))
    output_test = np.asarray(tests)
    output_test = output_test.reshape(output_length, number_of_classes)

    std_data = np.std(output_test[:, 0])
    mean_data = np.mean(output_test[:, 0])
    for count, item in enumerate(output_test):
        if item[0] <= mean_data + std_data:
            output_test[count][0] = int(0)
        else:
            output_test[count][0] = int(1)
            
    return output_test

def generate_unconditionally(random_seed=1):
    # Input:
    #   random_seed - integer
    
    # X and y data are dynaically created to minimize risk of running out of memorry 
    data = prep_data(strokes[:1])
    

    train(data)
    
    temp_test = test(get_windows(data, 0)[0])
    
    
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