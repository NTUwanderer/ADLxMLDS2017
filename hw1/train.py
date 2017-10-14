from __future__ import print_function

import random, collections, time, argparse
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.contrib import rnn
from random import shuffle

random.seed(1)

data_path = "./data/"
# parser = argparse.ArgumentParser()
# parser.add_argument("data_path", help="path to directory data")
# args = parser.parse_args()
# data_path = args.data_path

map1_table = pd.read_table(data_path + "/phones/48_39.map", sep="\t", header = None)

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

numOfFeatures = 69

train_dtype = {'frame': np.string_}
for i in range(numOfFeatures):
    train_dtype[i] = np.float64

label_dtype = {'frame': np.string_, 'label': np.string_}

train_col = list(range(numOfFeatures))
train_col.insert(0, 'frame')
label_col = ['frame', 'label']

train = pd.read_table(data_path + "/fbank/train.ark", sep=" ", header = None, names = train_col)
label = pd.read_table(data_path + "/label/train.lab", sep=",", header = None, names = label_col)

train_with_label = train.join(label.set_index('frame'), on='frame')
del train
del label

suffix="_1"

files = []
group = []
labels = []

for frame in train_with_label.values:
    frame = frame.tolist()
    if frame[0].endswith(suffix):
        if len(group) > 0:
            files.append(group)
            labels.append(label)

        group = np.zeros([n_input - 1, numOfFeatures]).tolist()
        label = []
    
    frame.pop(0)
    label.append(phoneToIndex[map1[frame.pop()]])
    group.append(frame)

del group
del label

numOfFiles = len(files)

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


# Target log path
logs_path = './'
writer = tf.summary.FileWriter(logs_path)


def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

# Parameters
learning_rate = 0.001
training_iters = 10

# number of units in RNN cell
n_hidden = 512

# tf Graph input
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

# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()
# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    writer.add_graph(session.graph)
    
    step = 0
    while step < training_iters:
        # Generate a minibatch. Add some randomness on selection process.
        sequence = list(range(numOfFiles))
        shuffle(sequence)

        for index in sequence:
            group = files[index]
            
            fbanks = []
            label = np.zeros([len(group) - n_input + 1, numOfPhones], dtype=float)
            for offset in range(0, len(group) - n_input + 1):
                fbank = [ group[i]  for i in range(offset, offset + n_input) ]
                fbanks.append(fbank)
                label[offset][labels[index][offset]] = 1
        
            #fbanks = np.reshape(np.array(fbanks), [-1, n_input, numOfFeatures])
            #label = np.reshape(label, [len(group), -1])
            

            _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={x: fbanks, y: label})
            loss_total += loss
            acc_total += acc

        print("Iter= " + str(step+1) + ", Average Loss= " + \
             "{:.6f}".format(loss_total/numOfFiles) + ", Average Accuracy= " + \
             "{:.2f}%".format(100*acc_total/numOfFiles))
        acc_total = 0
        loss_total = 0
        step += 1
    print("Optimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))
    print("Run on command line.")
    print("\ttensorboard --logdir=%s" % (logs_path))
    print("Point your web browser to: http://localhost:6006/")
    
    saver.save(session, "tmp/model.ckpt")


