import argparse

parser = argparse.ArgumentParser()
parser.add_argument("data_path", help="Path to data")
parser.add_argument("-t", "--test", action="store_true", help="Test rather than train")
parser.add_argument("-p", "--peer", action="store_true", help="Test the peer review videos")
parser.add_argument("--output_model_path", help="Where to store models of training")
parser.add_argument("-m", "--model_path", help="Path to model to retrain")
parser.add_argument("-o", "--output_path", help="Where to store predictions")
args = parser.parse_args()

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
# import ipdb
import time
# import cv2
import json
from keras.preprocessing import sequence
# import matplotlib.pyplot as plt

class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, n_video_lstm_step, n_caption_lstm_step, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.n_video_lstm_step=n_video_lstm_step
        self.n_caption_lstm_step=n_caption_lstm_step

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')
        #self.bemb = tf.Variable(tf.zeros([dim_hidden]), name='bemb')

        self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=True)
        self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=True)

        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def build_model(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step, self.dim_image])

        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_caption_lstm_step+1])
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_lstm_step+1])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b ) # (batch_size*n_lstm_steps, dim_hidden)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden])

        # state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        # state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        size1 = self.lstm1.state_size
        size2 = self.lstm2.state_size
        state1 = tf.split(tf.zeros([self.batch_size, size1[0] + size1[1]]), size1, axis=1)
        state2 = tf.split(tf.zeros([self.batch_size, size2[0] + size2[1]]), size2, axis=1)
        padding = tf.zeros([self.batch_size, self.dim_hidden])

        probs = []
        loss = 0.0

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            ##############################  Encoding Stage ##################################
            for i in range(0, self.n_video_lstm_step):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(image_emb[:,i,:], state1)

                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)

            ############################# Decoding Stage ######################################
            for i in range(0, self.n_caption_lstm_step): ## Phase 2 => only generate captions
                #if i == 0:
                #    current_embed = tf.zeros([self.batch_size, self.dim_hidden])
                #else:
                with tf.device("/cpu:0"):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])

                tf.get_variable_scope().reuse_variables()

                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(padding, state1)

                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)

                labels = tf.expand_dims(caption[:, i+1], 1)
                indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
                concated = tf.concat([indices, labels], 1)
                onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

                logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
                cross_entropy = cross_entropy * caption_mask[:,i]
                probs.append(logit_words)

                current_loss = tf.reduce_sum(cross_entropy)/self.batch_size
                loss = loss + current_loss

            return loss, video, caption, caption_mask, probs


    def build_generator(self):
        video = tf.placeholder(tf.float32, [1, self.n_video_lstm_step, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [1, self.n_video_lstm_step])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_video_lstm_step, self.dim_hidden])

        size1 = self.lstm1.state_size
        size2 = self.lstm2.state_size
        state1 = tf.split(tf.zeros([1, size1[0] + size1[1]]), size1, axis=1)
        state2 = tf.split(tf.zeros([1, size2[0] + size2[1]]), size2, axis=1)
        padding = tf.zeros([1, self.dim_hidden])

        generated_words = []

        probs = []
        embeds = []

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for i in range(0, self.n_video_lstm_step):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(image_emb[:, i, :], state1)

                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)

            for i in range(0, self.n_caption_lstm_step):
                tf.get_variable_scope().reuse_variables()

                if i == 0:
                    with tf.device('/cpu:0'):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int64))

                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(padding, state1)

                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)

                logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b)
                max_prob_index = tf.argmax(logit_words, 1)[0]
                generated_words.append(max_prob_index)
                probs.append(logit_words)

                with tf.device("/cpu:0"):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                    current_embed = tf.expand_dims(current_embed, 0)

                embeds.append(current_embed)

            return video, video_mask, generated_words, probs, embeds


#=====================================================================================
# Global Parameters
#=====================================================================================
video_path = args.data_path

video_train_feat_path = os.path.join(video_path, 'training_data/feat')
if args.peer:
    video_test_feat_path = os.path.join(video_path, 'peer_review/feat')
else:
    video_test_feat_path = os.path.join(video_path, 'testing_data/feat')

video_train_label_path = os.path.join(video_path, 'training_label.json')
# video_test_label_path = os.path.join(video_path, 'testing_label.json')

if args.peer:
    video_id_path = os.path.join(video_path, 'peer_review_id.txt')
else:
    video_id_path = os.path.join(video_path, 'testing_id.txt')

dict_path = 'dict'

#=======================================================================================
# Train Parameters
#=======================================================================================
dim_image = 4096
dim_hidden= 256

n_video_lstm_step = 80
# n_caption_lstm_step = 20
n_caption_lstm_step = 40
n_frame_step = 80

n_epochs = 40
batch_size = 32
learning_rate = 0.001

CURSOR_UP_ONE = '\033[F'
ERASE_LINE = '\033[K'

def get_video_data(video_label_path):
    with open(video_label_path, 'r') as f:
        captions = json.load(f)
    return captions
    
def preProBuildWordVocab(sentence_iterator, word_count_threshold=5):
    # borrowed this function from NeuralTalk
    print ('preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold))
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
           word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print ('filtered words from %d to %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '<pad>'
    ixtoword[1] = '<bos>'
    ixtoword[2] = '<eos>'
    ixtoword[3] = '<unk>'

    wordtoix = {}
    wordtoix['<pad>'] = 0
    wordtoix['<bos>'] = 1
    wordtoix['<eos>'] = 2
    wordtoix['<unk>'] = 3

    for idx, w in enumerate(vocab):
        wordtoix[w] = idx+4
        ixtoword[idx+4] = w

    word_counts['<pad>'] = nsents
    word_counts['<bos>'] = nsents
    word_counts['<eos>'] = nsents
    word_counts['<unk>'] = nsents

    bias_init_vector = np.array([1.0 * word_counts[ ixtoword[i] ] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range

    return wordtoix, ixtoword, bias_init_vector

def createDict():
    train_captions = get_video_data(video_train_label_path)

    captions_list = list([])
    v_ids = list([])
    
    for i, video in enumerate(train_captions):
        captions_list = captions_list + list(video['caption'])
        v_ids = v_ids + list([video['id']] * len(video['caption']))

    captions = np.asarray(captions_list, dtype=np.object)
    v_ids = np.asarray(v_ids, dtype=np.object)

    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)
    captions = map(lambda x: x.replace('"', ''), captions)
    captions = map(lambda x: x.replace('\n', ''), captions)
    captions = map(lambda x: x.replace('?', ''), captions)
    captions = map(lambda x: x.replace('!', ''), captions)
    captions = map(lambda x: x.replace('\\', ''), captions)
    captions = map(lambda x: x.replace('/', ''), captions)

    captions = list(captions)
    captions = np.asarray(captions, dtype=np.object)

    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold=3)

    # np.save('./' + dict_path + '/wordtoix', wordtoix)
    # np.save('./' + dict_path + '/ixtoword', ixtoword)
    # np.save('./' + dict_path + '/bias_init_vector', bias_init_vector)
    
    return v_ids, captions, wordtoix, ixtoword, bias_init_vector


def train():

    v_ids, captions, wordtoix, ixtoword, bias_init_vector = createDict()

    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(wordtoix),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step,
            bias_init_vector=bias_init_vector)

    tf_loss, tf_video, tf_caption, tf_caption_mask, tf_probs = model.build_model()
    print('tf_loss: ', tf_loss)
    sess = tf.InteractiveSession()
    # sess = tf.Session()
    
    # my tensorflow version is 0.12.1, I write the saver with version 1.0
    saver = tf.train.Saver(max_to_keep=100)
    # saver = tf.train.Saver()
    # with tf.variable_scope(tf.get_variable_scope(), reuse=False):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    tf.global_variables_initializer().run()

    #new_saver = tf.train.Saver()
    #new_saver = tf.train.import_meta_graph('./rgb_models/model-1000.meta')
    #new_saver.restore(sess, tf.train.latest_checkpoint('./models/'))

    loss_fd = open('loss.txt', 'w')
    loss_to_draw = []

    for epoch in range(0, n_epochs):
        loss_to_draw_epoch = []

        index = list(range(len(captions)))
        # np.random.shuffle(index)
        s_captions = captions[index]
        s_v_ids = v_ids[index]

        for start, end in zip(
                range(0, len(captions), batch_size),
                range(batch_size, len(captions), batch_size)):

            start_time = time.time()

            current_batch = s_captions[start:end]
            current_videos = s_v_ids[start:end]

            current_feats = np.zeros((batch_size, n_video_lstm_step, dim_image))
            current_feats_vals = list(map(lambda vid: np.load(os.path.join(video_train_feat_path, vid + '.npy')), current_videos))

            current_video_masks = np.zeros((batch_size, n_video_lstm_step))

            for ind,feat in enumerate(current_feats_vals):
                current_feats[ind][:len(current_feats_vals[ind])] = feat
                current_video_masks[ind][:len(current_feats_vals[ind])] = 1

            current_captions = list(map(lambda x: '<bos> ' + x, current_batch))

            for idx, each_cap in enumerate(current_captions):
                word = each_cap.lower().split(' ')
                if len(word) < n_caption_lstm_step:
                    current_captions[idx] = current_captions[idx] + ' <eos>'
                else:
                    new_word = ''
                    for i in range(n_caption_lstm_step-1):
                        new_word = new_word + word[i] + ' '
                    current_captions[idx] = new_word + '<eos>'

            current_caption_ind = []
            for cap in current_captions:
                current_word_ind = []
                for word in cap.lower().split(' '):
                    if word in wordtoix:
                        current_word_ind.append(wordtoix[word])
                    else:
                        current_word_ind.append(wordtoix['<unk>'])
                current_caption_ind.append(current_word_ind)

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=n_caption_lstm_step)
            current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix), 1] ) ] ).astype(int)
            current_caption_masks = np.zeros( (current_caption_matrix.shape[0], current_caption_matrix.shape[1]) )
            nonzeros = np.array( list(map(lambda x: (x != 0).sum() + 1, current_caption_matrix )) )

            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1

            # probs_val = sess.run(tf_probs, feed_dict={
            #     tf_video:current_feats,
            #     tf_caption: current_caption_matrix
            #     })

            _, loss_val = sess.run([train_op, tf_loss], feed_dict={
                            tf_video: current_feats,
                            tf_caption: current_caption_matrix,
                            tf_caption_mask: current_caption_masks
                            })
            loss_to_draw_epoch.append(loss_val)
            
            if start != 0:
                print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)
            print('idx: ', int(end / batch_size), "/", int(len(captions)/batch_size), " Epoch: ", epoch, " loss: ", loss_val, 'm loss: ', np.mean(loss_to_draw_epoch), ' Elapsed time: ', str((time.time() - start_time)))
            loss_fd.write('epoch ' + str(epoch) + ' loss ' + str(loss_val) + '\n')

        # draw loss curve every epoch
        # loss_to_draw.append(np.mean(loss_to_draw_epoch))
        # plt_save_dir = "./loss_imgs"
        # plt_save_img_name = str(epoch) + '.png'
        # plt.plot(range(len(loss_to_draw)), loss_to_draw, color='g')
        # plt.grid(True)
        # plt.savefig(os.path.join(plt_save_dir, plt_save_img_name))

        if np.mod(epoch, 1) == 0:
            print("Epoch ", epoch, " is done. Saving the model ...")
            saver.save(sess, args.output_model_path, global_step=epoch)

    loss_fd.close()

def test(model_path='./models/model-100'):

    # test_captions = get_video_data(video_test_label_path)
    
    test_videos = []
    # for video in test_captions:
    #     test_videos.append(video['id'])
    f = open(video_id_path, 'r')
    for line in f:
        test_videos.append(line[:-1])

    _, __, w, i, b = createDict()

    ixtoword = pd.Series(i)

    bias_init_vector = b

    # ixtoword = pd.Series(np.load('./' + dict_path + 'ixtoword.npy').tolist())
    # bias_init_vector = np.load('./' + dict_path + 'bias_init_vector.npy')

    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step,
            bias_init_vector=bias_init_vector)

    video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()

    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    test_output_txt_fd = open(args.output_path, 'w')
    for idx, video_feat_path in enumerate(test_videos):
        video_feat = np.load(os.path.join(video_test_feat_path, video_feat_path + '.npy'))[None, ...]
        #video_feat = np.load(video_feat_path)
        #video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
        video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))

        generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
        generated_words = ixtoword[generated_word_index]

        punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
        generated_words = generated_words[:punctuation]

        generated_sentence = ' '.join(generated_words)
        generated_sentence = generated_sentence.replace('<bos> ', '')
        generated_sentence = generated_sentence.replace(' <eos>', '')
        generated_sentence = generated_sentence.replace(' <unk>', '')
        generated_sentence = generated_sentence.replace('<unk>', '')
        if idx != 0:
            test_output_txt_fd.write('\n')
        test_output_txt_fd.write(video_feat_path + ',')
        test_output_txt_fd.write(generated_sentence)

def main():
    if args.test:
        print ("Testing")
        test(args.model_path)
    else:
        print ("Training")
        train()

if __name__=="__main__":
    main()
