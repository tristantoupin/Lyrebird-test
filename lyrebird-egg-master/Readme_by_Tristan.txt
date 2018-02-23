Hi, and welcome to my Lyrebird project!

Here's some information you need to have to check my work. The test it, run open the notebooks/results.ipynb file and run it! To see high level logic, go in model/dummy.py. The really understand how everything works, take a look at model/part_1.py.


# - PART I -

##Overall NN:
The model was trained with the following parameters:
    - epochs = 100
    - learning_rate = 0.001
    - batch_size = 100 # number of windows in a single batch
    - hidden_layer_size = 400 # Single hidden layer of 400 cells
    - number_of_layers = 1
    - dropout_rate = 0.5
    - number_of_classes = 3
    - gradient_clip_margin = 4
    - window_size = 57


##Training techniques:
The training set I created is basically a very long strokes (concatenation of all strokes) and slipted this in windows. Windows a frames that move horizontally through the long stroke. The goal is to input several points at once. The size of the window (57) was selected based on Alex Graves' paper provided with the test. Some issues were found using this testing technique (see the Furthur imporvements section).


##Testing techniques:
I took a long time to decide how I should start a stroke. The first idea I had was to provide a window and generate the output by consecutively appending the previous output the new test window. Repeating this process would allow to generate strokes. 

Then I explored another idea; creating a probabilistic model to start the stroke. By training the model using each stroke with a padding at the begining. Then by looking where the trainning strokes most often starts, I could create a window filled with zeros but the last row. The last row would be the most probable coordinates to start based. I could introduce randomness by changing teh "most" probable coordinate by probabilities. This is a valuable idea to investigate in my opinion, but I went with the first idea to simplify the project.

NOTE: Each stroke generated are 700 dots long.


##Furthur imporvements:
The main concern the I have with my model is the lack of testing. Letters look like letters but are not precise not clear for the most part. I believe that training over the whole data set would of make a huge difference. I do not want the model to overfit the data since I want it to "imagine" the next letters. Therefore the number of epoch and learning rate seems decent here.

Another issues I experience while testing was that the sequece of words outputted are not alligned. In other words, the tests output have words that are higher than others on the same line. Considering all the strokes as a single very long stroke caused this problem. I assumed that all stroke would be relatively closer next to each other. To correct this problem, I could either creating a function normalizing the strokes to stay inside bounds (min and max) but this might cause the the letters to be deformed.


##Remarks:
(1) I remarked that numbers are poorly recreated by the model. Probably because There are very few examples of numbers in the data set. try --> stroke = generate_unconditionally(4) | plot_stroke(stroke)
(2) It was my first time working with LSTM cells in Tensorflow. I have experience with LSTMs in Keras and in theory and also with TensorFlow.