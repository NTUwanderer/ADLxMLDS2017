from __future__ import print_function

import random, collections, time, argparse

numOfPhones = 39

parser = argparse.ArgumentParser()
# parser.add_argument("data_path", help="path to directory data")
parser.add_argument('-f', '--feature', default="fbank", choices = ['fbank', 'mfcc'], help="default fbank")
parser.add_argument('-n', '--num_steps', default=5, type=int, help="set num_steps to truncate")
parser.add_argument('-m', '--model_path', default="./tmp/model.ckpt", help="write model to path")
parser.add_argument('-c', '--n_hidden', default=numOfPhones, type=int, help="n_hidden in LSTM")
parser.add_argument('-r', '--rnn_cell', default="rnn", choices = ['rnn', 'lstm', 'gru'], help="Which basic cell")
parser.add_argument('-e', '--epoch', default=10, type=int, help="num of epoch")

import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.contrib import rnn
from random import shuffle

random.seed(1)

data_path = "./data/"


args = parser.parse_args()
feature = args.feature
num_steps = args.num_steps
model_path = args.model_path
n_hidden = args.n_hidden
rnn_cell = args.rnn_cell
epoch = args.epoch

first_half_num = int(num_steps / 2)
# first_half_num = 0
second_half_num = num_steps - first_half_num

map1_table = pd.read_table(data_path + "/phones/48_39.map", sep="\t", header = None)

map1 = dict()
phoneToIndex = dict()
indexToPhone = []


counter = 0
for trans in map1_table.values:
    map1[trans[0]] = trans[1]
    if trans[1] not in phoneToIndex:
        phoneToIndex[trans[1]] = counter
        indexToPhone.append(trans[1])
        counter += 1

if feature == 'fbank':
    numOfFeatures = 69
    train_path = '/fbank/train.ark'
else:
    numOfFeatures = 39
    train_path = '/mfcc/train.ark'

train_col = list(range(numOfFeatures))
train_col.insert(0, 'frame')
label_col = ['frame', 'label']

train = pd.read_table(data_path + train_path, sep=" ", header = None, names = train_col)
label = pd.read_table(data_path + "/label/train.lab", sep=",", header = None, names = label_col)

train_with_label = train.join(label.set_index('frame'), on='frame')
del train
del label

suffix="_1"

files = []
group = []
labels = []

def remainSize(num_steps, size):
    return size - size % num_steps

total_length = 0
for frame in train_with_label.values:
    frame = frame.tolist()
    if frame[0].endswith(suffix):
        if len(group) > 0:
            total_length += len(group) - num_steps + 1
            files.append(group)
            labels.append(label)

        frame.pop(0)
        l = phoneToIndex[map1[frame.pop()]]

        group = [frame] * num_steps
        label = [l] * num_steps
        continue
    
    frame.pop(0)
    label.append(phoneToIndex[map1[frame.pop()]])
    group.append(frame)

del group
del label

files = np.array(files)
labels = np.array(labels)

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

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

# Parameters
learning_rate = 0.0001

# number of units in RNN cell
batch_size = None
# tf Graph input
x = tf.placeholder("float", [batch_size, num_steps, numOfFeatures], name="input_placeholder")
y = tf.placeholder("int32", [batch_size, num_steps], name="labels_placeholder")
#init_state = tf.zeros([batch_size, n_hidden])
with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [n_hidden, numOfPhones])
    b = tf.get_variable('b', [numOfPhones], initializer=tf.constant_initializer(0.0))

#cell = tf.contrib.rnn.BasicRNNCell(n_hidden)
if rnn_cell == 'gru':
    cell = rnn.MultiRNNCell([rnn.GRUCell(n_hidden),rnn.GRUCell(n_hidden)])
elif rnn_cell == 'lstm':
    cell = rnn.MultiRNNCell([rnn.LSTMCell(n_hidden, state_is_tuple=True),rnn.LSTMCell(n_hidden, state_is_tuple=True)], state_is_tuple=True)
else:
    cell = rnn.MultiRNNCell([rnn.BasicRNNCell(n_hidden),rnn.BasicRNNCell(n_hidden)])

rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

logits = tf.reshape(
            tf.matmul(tf.reshape(rnn_outputs, [-1, n_hidden]), W) + b,
            [-1, num_steps, numOfPhones])
#[-1, num_steps, numOfPhones])

pred = tf.nn.softmax(logits)
pred1, pred2 = tf.split(pred, [first_half_num, second_half_num], 1)

trueLabel = tf.one_hot(y, numOfPhones, on_value=1.0, off_value=0.0)
logits1, logits2 = tf.split(logits, [first_half_num, second_half_num], 1)
trueLabel1, trueLabel2 = tf.split(trueLabel, [first_half_num, second_half_num], 1)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=trueLabel2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred2,2), tf.argmax(trueLabel2,2))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()
# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    acc_total = 0
    loss_total = 0

    step = 0
    while step < epoch:
        # Generate a minibatch. Add some randomness on selection process.
        sequence = list(range(total_length))
        shuffle(sequence)

        shuffledFiles = np.zeros([total_length, num_steps, numOfFeatures])
        shuffledLabels = np.zeros([total_length, num_steps])
        counter = 0
        for index in range(numOfFiles):
            group = files[index]
            label = labels[index]
            for i in range(len(group) - num_steps + 1):
                newIndex = sequence[counter]
                shuffledFiles[newIndex] = group[i:i+num_steps]
                shuffledLabels[newIndex] = label[i:i+num_steps]
                counter += 1
                
        tempBatchSize = 128

        count = 0

        rnd = int(shuffledFiles.shape[0] / tempBatchSize)
        while (count+1) <= rnd:
            fbanks = shuffledFiles[count * tempBatchSize:(count+1) * tempBatchSize]
            label = shuffledLabels[count * tempBatchSize:(count+1) * tempBatchSize]

            count += 1

            _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, logits], \
                                                feed_dict={x: fbanks, y: label})
            loss_total += loss
            acc_total += acc

            print("Iter= " + str(step) + ", round= " + str(count) + "/" + str(rnd) + ", Average Loss= " + \
                 "{:.6f}".format(loss_total/(count+1)) + ", Average Accuracy= " + \
                 "{:.2f}%".format(100*acc_total/(count+1)) + ", Loss= " + \
                 "{:.6f}".format(loss) + ", Accuracy= " + \
                 "{:.2f}%".format(100*acc))

        acc_total = 0
        loss_total = 0
        step += 1
    print("Optimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))
    
    saver.save(session, model_path)

