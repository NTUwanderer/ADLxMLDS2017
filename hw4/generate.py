import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import DCGAN
import skimage.io
import skimage.transform
import os
from skip_thoughts import configuration
from skip_thoughts import encoder_manager
import numpy as np
import scipy.spatial.distance as sd
import pdb
import random
import pickle

import argparse

np.random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--testing_text', type=str, default='data/testing_text.txt', help='path to testing_text.txt')
parser.add_argument('-m', '--model', type=str, help='Model name')
parser.add_argument('-p', '--path', type=str, default='dcgan-model-final', help='Model path')
parser.add_argument('-o', '--output', type=str, default='samples', help='samples path')
args = parser.parse_args()

MODEL_PATH="./skip_thoughts/unidirectional/"
VOCAB_FILE = MODEL_PATH + "vocab.txt"
EMBEDDING_MATRIX_FILE = MODEL_PATH + "embeddings.npy"
CHECKPOINT_PATH = MODEL_PATH + "model.ckpt-501424"


num_epochs = 500
learning_rate = 0.0001
image_dir = 'data/faces/'
trim_tags = 'trim.txt'
dis_updates = 1
gen_updates = 2
num_images = 5
#model_path = 'dcgan-model'
model_path = args.path
#pic_save_path = 'samples/'
pic_save_path = args.output + '/'
early_text = 'green hair blue eyes'
#test_txt = 'data/testing_text.txt'
test_txt = args.testing_text

params = dict(
    z_dim = 250,
    t_dim = 256,
    batch_size = 5,
    image_size = 64,
    gf_dim = 64,
    df_dim = 64,
    gfc_dim = 1024,
    dfc_dim = 1024,
    istrain = False,
    caption_length = 2400
)


gan = DCGAN.GAN(params)
_, _, _, _ = gan.build_model()

sess = tf.InteractiveSession()
saver = tf.train.Saver()
#ckpt = tf.train.get_checkpoint_state(model_path)
#print ('path: ', ckpt.model_checkpoint_path)
#saver.restore(sess , ckpt.model_checkpoint_path)

model_names = ['model-1080', 'model-1130', 'model-1340', 'model-1560', 'model-1620']

for i_th, name in enumerate(model_names):

    model_name = model_path + '/' + name
    print ('model_name: ', model_name)
    saver.restore(sess , model_name)
    input_tensors, outputs = gan.build_generator()
    
    
    ids = []
    tags = []
    
    f = open(test_txt , 'r')
    for line in f.readlines():
        id = line.split(',')[0]        
        tag = line.split(',')[1]
        ids.append(id)
        tags.append(tag)
    
    z_noise = np.random.uniform(-1, 1, [len(ids), num_images, params['z_dim']])
    #encodings = encoder.encode(tags)
    #encodings = np.array(encodings)
    #print(np.shape(encodings))
    #encodings = np.expand_dims(encodings , axis=1)
    #print(np.shape(encodings))
    #caption = np.tile(encodings, (1,num_images,1))
    #print(np.shape(caption))
    
    caption = pickle.load(open('cap_test.pkl' , 'rb'))
    
    #caption = np.reshape(caption , (-1, num_images , params['caption_length']))
    
    caption_images = {}
    for i in range(0,len(ids)):
        caption_images[ids[i]] =\
        sess.run([outputs['generator']],
                feed_dict = {input_tensors['t_real_caption'] : caption[i],
                            input_tensors['t_z'] : z_noise[i]}) # (1,num_images,64,64,3)
    
    #pdb.set_trace()
    
    for i in range(len(ids)):
        skimage.io.imsave(os.path.join(pic_save_path , 'sample_' + ids[i] + '_' + str(i_th+1) + '.jpg') , caption_images[ids[i]][0][0])
