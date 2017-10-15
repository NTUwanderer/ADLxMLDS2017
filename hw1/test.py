import random, collections, time, argparse
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.contrib import rnn
from random import shuffle

data_path = "./data/"
output_file = "./output/myFirstOutput"

# parser = argparse.ArgumentParser()
# parser.add_argument("data_path", help="path to directory data")
# args = parser.parse_args()
# data_path = args.data_path

map1_table = pd.read_table(data_path + "/phones/48_39.map", sep="\t", header = None)
map2_table = pd.read_table(data_path + "/48phone_char.map", sep="\t", header = None)

map1 = dict()
phoneToIndex = dict()
indexToPhone = []

numOfPhones = 39
n_input = 3

counter = 0
for trans in map1_table.values:
    map1[trans[0]] = trans[1]
    if trans[1] not in phoneToIndex:
        phoneToIndex[trans[1]] = counter
        indexToPhone.append(trans[1])
        counter += 1

map2 = dict()
for trans in map2_table.values:
    map2[trans[0]] = trans[2]

numOfFeatures = 69

train_dtype = {'frame': np.string_}
for i in range(numOfFeatures):
    train_dtype[i] = np.float64

test_col = list(range(numOfFeatures))
test_col.insert(0, 'frame')

test = pd.read_table(data_path + "/fbank/test.ark", sep=" ", header = None, names = test_col)

suffix="_1"

files = []
group = []
frameIds = []

for frame in test.values:
    frame = frame.tolist()
    if frame[0].endswith(suffix):
        if len(group) > 0:
            files.append(group)
            frameIds.append(frame[0][:-2])

        group = np.zeros([n_input - 1, numOfFeatures]).tolist()
        label = []
    
    frame.pop(0)
    group.append(frame)

del group

tf.reset_default_graph()

learning_rate = 0.001
n_hidden = 512
x = tf.placeholder("float", [None, n_input, numOfFeatures])
y = tf.placeholder("float", [None, numOfPhones])

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, numOfPhones]))
}
biases = {
    'out': tf.Variable(tf.random_normal([numOfPhones]))
}

def RNN(x, weights, biases):

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input * numOfFeatures])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,n_input * numOfFeatures,1)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)
# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()

def myTrim(onehot_pred_index, threshold = 4):
    trimmed_pred_index = []
    sil = 7
    current = 7

    temp = 7
    temp_count = 0
    for index in onehot_pred_index:
        if index == temp:
            if temp == current:
                continue
            
            temp_count += 1
            if temp_count == 4:
                trimmed_pred_index.append(temp)
                current = temp

        else:
            temp_count = 1
            temp = index

    if trimmed_pred_index[-1] == sil:
        trimmed_pred_index.pop()

    return trimmed_pred_index

outputCSV = pd.DataFrame(columns=['id','phone_sequence'])

with tf.Session() as session:
    saver.restore(session, 'tmp/model.ckpt')

    for i in range(len(files)):
        print(str(i) + "/" + str(len(files)))
        fbanks = []
        group = files[i]
        for offset in range(0, len(group) - n_input + 1):
            fbank = [ group[i]  for i in range(offset, offset + n_input) ]
            fbanks.append(fbank)
    
        onehot_pred = session.run(pred, feed_dict={x: fbanks})
        onehot_pred_index = tf.argmax(onehot_pred, 1).eval()

        trimmed_pred_index = myTrim(onehot_pred_index)
    
        phone_pred = ''.join([ map2[indexToPhone[index]]  for index in trimmed_pred_index ])

        outputCSV.loc[i] = [frameIds[i], phone_pred]

outputCSV.to_csv(path_or_buf=myFirstOutput)

