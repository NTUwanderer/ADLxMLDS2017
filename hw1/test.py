import random, collections, time, argparse

numOfPhones = 39

parser = argparse.ArgumentParser()
# parser.add_argument("data_path", help="path to directory data")
parser.add_argument('-f', '--feature', default="fbank", choices = ['fbank', 'mfcc'], help="default fbank")
parser.add_argument('-n', '--num_steps', default=5, type=int, help="set num_steps to truncate")
parser.add_argument('-m', '--model_path', default="./tmp/model.ckpt", help="read model from path")
parser.add_argument('-c', '--n_hidden', default=numOfPhones, type=int, help="n_hidden in LSTM")
parser.add_argument('-o', '--output_path', default="./output/output.ckpt", help="output csv path")

import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.contrib import rnn
from random import shuffle

data_path = "./data/"

args = parser.parse_args()
feature = args.feature
num_steps = args.num_steps
model_path = args.model_path
n_hidden = args.n_hidden
output_path = args.output_path

map1_table = pd.read_table(data_path + "/phones/48_39.map", sep="\t", header = None)
map2_table = pd.read_table(data_path + "/48phone_char.map", sep="\t", header = None)

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

map2 = dict()
for trans in map2_table.values:
    map2[trans[0]] = trans[2]

if feature == 'fbank':
    numOfFeatures = 69
    test_path = '/fbank/test.ark'
else:
    numOfFeatures = 39
    test_path = '/mfcc/test.ark'

train_dtype = {'frame': np.string_}
for i in range(numOfFeatures):
    train_dtype[i] = np.float64

test_col = list(range(numOfFeatures))
test_col.insert(0, 'frame')

test = pd.read_table(data_path + test_path, sep=" ", header = None, names = test_col)

suffix="_1"

files = []
group = []
frameIds = []
num_steps=10

def remainSize(num_steps, size):
    return size - size % num_steps

for frame in test.values:
    frame = frame.tolist()
    if frame[0].endswith(suffix):
        if len(group) > 0:
            rSize = remainSize(num_steps, len(group))
            files.append(group[:rSize])

        frameIds.append(frame[0][:-2])

        group = []
    
    frame.pop(0)
    group.append(frame)

if len(group) > 0:
    rSize = remainSize(num_steps, len(group))
    files.append(group[:rSize])

del group

files = np.array(files)

tf.reset_default_graph()

learning_rate = 0.001
batch_size = None
x = tf.placeholder("float", [batch_size, num_steps, numOfFeatures], name="input_placeholder")
y = tf.placeholder("int32", [batch_size, num_steps], name="labels_placeholder")
#init_state = tf.zeros([batch_size, n_hidden])
with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [n_hidden, numOfPhones])
    b = tf.get_variable('b', [numOfPhones], initializer=tf.constant_initializer(0.0))

#cell = tf.contrib.rnn.BasicRNNCell(n_hidden)
cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])
rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

logits = tf.reshape(
            tf.matmul(tf.reshape(rnn_outputs, [-1, n_hidden]), W) + b,
            [-1, num_steps, numOfPhones])
#[-1, num_steps, numOfPhones])

pred = tf.nn.softmax(logits)

trueLabel = tf.one_hot(y, numOfPhones, on_value=1.0, off_value=0.0)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=trueLabel))
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(trueLabel,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()

def myTrim(onehot_pred_index, threshold = 2):
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
    saver.restore(session, model_path)

    for i in range(len(files)):
        print(str(i) + "/" + str(len(files)))
        fbanks = []
        group = np.array(files[i])
        size = group.shape[0]
        fbanks = np.reshape(group, [int(size / num_steps), num_steps, numOfFeatures])
    
        onehot_pred = session.run(pred, feed_dict={x: fbanks})
        onehot_pred = np.reshape(onehot_pred, [-1, numOfPhones])
        onehot_pred_index = tf.argmax(onehot_pred, 1).eval()

        trimmed_pred_index = myTrim(onehot_pred_index)
    
        phone_pred = ''.join([ map2[indexToPhone[index]]  for index in trimmed_pred_index ])

        outputCSV.loc[i] = [frameIds[i], phone_pred]

outputCSV.to_csv(path_or_buf=output_path, index=False)

