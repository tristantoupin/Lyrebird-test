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

'''
Check part_1.py for all helper methods (same folder)
'''
def generate_unconditionally(random_seed=6):    
    # restore model and setup variables
    prep_model("model_v1")
    
    '''
    slipt_data_part_1 creates data sets where 0.8 is 80-20 
    train/test and 80 here is the maximum number of strokes
    used to train the model. I used only 80 strokes since
    I was not able to train the model fast enough on a 
    single CPU to test the model. The results are still decent.
    '''
    X_train, X_test, y_train, y_test = slipt_data_part_1(strokes, random_seed, 0.8, 80)
    
    '''
    Print the shapes of data sets
    '''
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    
    '''
    Train the model
    '''
#     train(X_train, y_train, "model_v2")
    
    '''
    generate the test output based on the seed
    print the expected results (y_test)
    print the calculated results (X_test)
    '''
    test_output, output_start = test(X_test)
    print("Generated stroke:")
    
    return (test_output)


'''
As specified in my email, part 2 and 3 are
coming soon.
'''

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