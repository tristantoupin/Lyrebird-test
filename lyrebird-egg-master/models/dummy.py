import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm_notebook
from part_1 import *
from numpy import random
from utils import plot_stroke
# from part_2 import *
# from part_3 import *

strokes = np.load('../data/strokes.npy')
strings_path = "../data/sentenses.txt"
stroke = strokes[0]



def generate_unconditionally(random_seed=1):
    # Input:
    #   random_seed - integer
    prep_model("model_v1")
    
    X_train, X_test, y_train, y_test = slipt_data_part_1(strokes, random_seed, 0.8, 80)
    
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    train(X_train, y_train, "model_256_b-100")

    test_output, output_start = test(X_test)
    print("Example")
    plot_stroke(y_test[output_start:output_start+700])
    print("Generated")

    # Output:
    #   stroke - numpy 2D-array (T x 3)
    return (test_output)


def prep_data_part2(train_ratio, max_train_size = float("inf")):
    # the variable max_train_size, was needed in order to reduce the training data set (time=
    data = ""
    with open ("strings_path.txt", "r") as myfile:
        data = myfile.readlines()
        
    assert len(data) ==  len(strokes)
    
    train_length = min(max_train_size, int(train_ratio * len(data)))
    
    X_train = data[:train_length]
    X_test = data[train_length:]
    y_train = data[:train_length]
    y_test = data[train_length:]
    
    return X_train, X_test, y_train, y_test

    



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