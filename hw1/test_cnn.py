import random, collections, time, argparse

numOfPhones = 39

parser = argparse.ArgumentParser()
parser.add_argument("data_path", help="path to directory data")
parser.add_argument('-f', '--feature', default="fbank", choices = ['fbank', 'mfcc', 'both'], help="default fbank")
parser.add_argument('-n', '--num_steps', default=5, type=int, help="set num_steps to truncate")
parser.add_argument('-m', '--model_path', default="./tmp/model.ckpt", help="read model from path")
parser.add_argument('-c', '--n_hidden', default=100, type=int, help="n_hidden in LSTM")
parser.add_argument('-o', '--output_path', default="./output/output.csv", help="output csv path")
parser.add_argument('-r', '--rnn_cell', default="rnn", choices = ['rnn', 'lstm', 'gru'], help="Which basic cell")
parser.add_argument('-l', '--n_layers', default=2, type=int, help="num of layers")
parser.add_argument('-d', '--dropout', default=0.1, type=float, help="um of layers")
parser.add_argument('-g', '--gpu', action="store_true", help="Use gpu0 instead of cpu0")

import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.contrib import rnn
from random import shuffle

#data_path = "./data/"

args = parser.parse_args()
data_path = args.data_path
feature = args.feature
num_steps = args.num_steps
model_path = args.model_path
n_hidden = args.n_hidden
rnn_cell = args.rnn_cell
output_path = args.output_path
n_layers = args.n_layers
dropout = args.dropout
if args.gpu:
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"

first_half_num = int(num_steps / 2)
# first_half_num = 0
second_half_num = num_steps - first_half_num

tf.reset_default_graph()

learning_rate = 0.0001

if feature == 'fbank' or feature == 'mfcc':
    if feature == 'fbank':
        numOfFeatures = 69
    else:
        numOfFeatures = 39
else:
    numOfFeatures = 108

batch_size = 128
with tf.device(device_name):
    x = tf.placeholder(tf.float32, [batch_size, num_steps, numOfFeatures], name="input_placeholder")
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name="labels_placeholder")
    #init_state = tf.zeros([batch_size, n_hidden])
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [n_hidden, numOfPhones])
        b = tf.get_variable('b', [numOfPhones], initializer=tf.constant_initializer(0.0))
    
    x_ = tf.reshape(x, [batch_size, num_steps, numOfFeatures, 1])
    
    tempW = tf.Variable(tf.random_normal([1, 3, 1, 64], stddev=1e-3), dtype=tf.float32)
    tempB = tf.Variable(tf.random_normal([64], stddev=1e-3), dtype=tf.float32)
    conv1 = tf.nn.conv2d(
            input=x_,
            filter=tempW,
            strides=[1, 1, 3, 1],
            padding='VALID'
            )
    conv1_relu = tf.nn.relu(conv1 + tempB)
    conv1_relu_r = tf.reshape(conv1_relu, [batch_size, num_steps, int(numOfFeatures/3) * 64])
    cnn_output = tf.layers.dense(
            inputs=conv1_relu_r,
            units=int(numOfFeatures/3)
            )

    #cell = tf.contrib.rnn.BasicRNNCell(n_hidden)
    if rnn_cell == 'gru':
        cell = rnn.MultiRNNCell([rnn.GRUCell(n_hidden)  for i in range(n_layers)])
        cell2 = rnn.MultiRNNCell([rnn.GRUCell(n_hidden)  for i in range(n_layers)])
    elif rnn_cell == 'lstm':
        cell = rnn.MultiRNNCell([rnn.LSTMCell(n_hidden, state_is_tuple=True)  for i in range(n_layers)], state_is_tuple=True)
        cell2 = rnn.MultiRNNCell([rnn.LSTMCell(n_hidden, state_is_tuple=True)  for i in range(n_layers)], state_is_tuple=True)
    else:
        cell = rnn.MultiRNNCell([rnn.BasicRNNCell(n_hidden)  for i in range(n_layers)])
        cell2 = rnn.MultiRNNCell([rnn.BasicRNNCell(n_hidden)  for i in range(n_layers)])
    
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout)
    cell2 = tf.contrib.rnn.DropoutWrapper(cell2, output_keep_prob=1.0 - dropout)
    rnn_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell, cell2, cnn_output, dtype=tf.float32)
    
    outputs1 = rnn_outputs[0]
    outputs2 = rnn_outputs[1]
    logits1 = tf.reshape(
                tf.matmul(tf.reshape(outputs1, [-1, n_hidden]), W) + b,
                [-1, num_steps, numOfPhones])
    
    logits2 = tf.reshape(
                tf.matmul(tf.reshape(outputs2, [-1, n_hidden]), W) + b,
                [-1, num_steps, numOfPhones])
    #[-1, num_steps, numOfPhones])
    #half_logits1, _1 = tf.split(logits1, [first_half_num, second_half_num], 1)
    #_2, half_logits2 = tf.split(logits2, [first_half_num, second_half_num], 1)
    #logits = tf.concat([half_logits1, half_logits2], 1)
    logits = tf.add(logits1, logits2)
    
    pred = tf.nn.softmax(logits)
    
    trueLabel = tf.one_hot(y, numOfPhones, on_value=1.0, off_value=0.0)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=trueLabel))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    #optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Model evaluation
    correct_pred = tf.equal(tf.argmax(pred,2), tf.argmax(trueLabel,2))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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

if feature == 'fbank' or feature == 'mfcc':
    if feature == 'fbank':
        test_path = '/fbank/test.ark'
    else:
        test_path = '/mfcc/test.ark'

    test_col = list(range(numOfFeatures))
    test_col.insert(0, 'frame')
    
    test = pd.read_table(data_path + test_path, sep=" ", header = None, names = test_col)

else:
    test_path = '/fbank/test.ark'
    test_path2 = '/mfcc/test.ark'
    test_col = list(range(69))
    test_col.insert(0, 'frame')
    test_col2 = list(range(39))
    test_col2.insert(0, 'frame')
    test = pd.read_table(data_path + test_path, sep=" ", header = None, names = test_col)
    test2 = pd.read_table(data_path + test_path2, sep=" ", header = None, names = test_col2)
    test = pd.merge(test, test2, on='frame')
    del test2

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
        # frame.pop(0)
        # group = [frame] * num_steps
        group = []
        # continue

    frame.pop(0)
    group.append(frame)

if len(group) > 0:
    files.append(group)

del test
del group

files = np.array(files)


CURSOR_UP_ONE = '\033[F'
ERASE_LINE = '\033[K'

saver = tf.train.Saver()

def myTrim(onehot_pred_index, threshold = 3):
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

        else:
            temp_count = 1
            temp = index

        if temp_count == threshold:
            trimmed_pred_index.append(temp)
            current = temp

    if trimmed_pred_index[-1] == sil:
        trimmed_pred_index.pop()

    return trimmed_pred_index

outputCSV = pd.DataFrame(columns=['id','phone_sequence'])

with tf.Session() as session:
    saver.restore(session, model_path)

    lenCounter = 0
    for i in range(len(files)):
        if i != 0:
            print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)
        print(str(i + 1) + "/" + str(len(files)))
        group = np.array(files[i])
        fbanks = np.zeros([len(group) - num_steps + 1, num_steps, numOfFeatures])
        for j in range(len(group) - num_steps + 1):
            fbanks[j] = np.array(group[j:j+num_steps])

        onehot_pred = np.zeros([0, num_steps, numOfPhones])
        while fbanks.shape[0] > 0:
            if fbanks.shape[0] > batch_size:
                two_parts = np.split(fbanks, [batch_size, fbanks.shape[0]])
                temp_fbanks = two_parts[0]
                fbanks = two_parts[1]
                del two_parts
                temp_onehot_pred = session.run(pred, feed_dict={x: temp_fbanks})

            else:
                origSize = fbanks.shape[0]
                repeatSize = int(batch_size / fbanks.shape[0]) + 1
                fbanks = np.repeat(fbanks, repeatSize, axis=0)
                temp_fbanks = np.split(fbanks, [batch_size, fbanks.shape[0]])[0]
                fbanks = np.zeros([0, num_steps, numOfFeatures])
                temp_onehot_pred = session.run(pred, feed_dict={x: temp_fbanks})

                temp_onehot_pred = np.split(temp_onehot_pred, [origSize, batch_size])[0]

            onehot_pred = np.concatenate((onehot_pred, temp_onehot_pred))

        myPred = np.zeros([len(group), numOfPhones])
        for index in range(onehot_pred.shape[0]):
            for j in range(num_steps):
                for k in range(numOfPhones):
                    myPred[index + j][k] += onehot_pred[index][j][k]
        onehot_pred_index = tf.argmax(myPred, 1).eval()

        trimmed_pred_index = myTrim(onehot_pred_index)
    
        phone_pred = ''.join([ map2[indexToPhone[index]]  for index in trimmed_pred_index ])

        lenCounter += len(phone_pred)
        outputCSV.loc[i] = [frameIds[i], phone_pred]

    print('lenCounter: ', lenCounter, ', average: ', lenCounter / len(files))
outputCSV.to_csv(path_or_buf=output_path, index=False)

