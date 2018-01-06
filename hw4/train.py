import tensorflow as tf
import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'

import DCGAN
import skimage.io
import skimage.transform
import numpy as np
import scipy.spatial.distance as sd
import pdb
import random
import pickle

from skip_thoughts import configuration
from skip_thoughts import encoder_manager

MODEL_PATH="./skip_thoughts/unidirectional/"
VOCAB_FILE = MODEL_PATH + "vocab.txt"
EMBEDDING_MATRIX_FILE = MODEL_PATH + "embeddings.npy"
CHECKPOINT_PATH = MODEL_PATH + "model.ckpt-501424"

"""
z_dim: noise dimension
t_dim: text  dimension
gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
caption_length: caption vector length
"""

def get_training_batch(batch_no, batch_size, image_size, z_dim,
                       caption_vector_length, images , embeddings):
    real_images = np.zeros((batch_size, 64, 64, 3))
    wrong_images = np.zeros((batch_size, 64, 64, 3))
    captions = np.zeros((batch_size, caption_vector_length))
    wrong_captions = np.zeros((batch_size, caption_vector_length))

    cnt = 0
    image_files = []
    for i in range(batch_no * batch_size, batch_no * batch_size + batch_size):
        idx = i % len(images)
        
        real_images[cnt] = images[idx]
        captions[cnt] = embeddings[idx]

        # Improve this selection of wrong image
        while True:
            wrong_image_id = random.randint(0, len(images)-1)
            if tags[idx] != tags[wrong_image_id]:
                wrong_images[cnt] = images[wrong_image_id]
                break

        while True:
            wrong_caption_id = random.randint(0, len(images)-1)
            if tags[idx] != tags[wrong_caption_id]:
                wrong_captions[cnt] = embeddings[wrong_caption_id]
                break

        #image_files.append(image_file)
        cnt += 1

    z_noise = np.random.uniform(-1.5, 1.5, [batch_size, z_dim])
    return real_images, wrong_images, captions, z_noise, wrong_captions



def read_images(image_dir , trim_tags , batch_size):
    
    images = []
    tmp_tags = []

    encoder = encoder_manager.EncoderManager()
    encoder.load_model(configuration.model_config(),
                    vocabulary_file=VOCAB_FILE,
                    embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                    checkpoint_path=CHECKPOINT_PATH)
    f = open(trim_tags , 'r')

    for line in f.readlines():
        splits = line.split(',')
        id = splits[0]
        if len(splits) <= 1 or len(splits[1]) <= 1:
            continue

        tags = splits[1]

        tmp_tags.append(tags)

        img = skimage.io.imread(os.path.join(image_dir , id + '.jpg'))
        img_resized = skimage.transform.resize(img, (64,64),mode='constant')
        images.append(img_resized)

    # captions = pickle.load(open('cap_b.pkl' , 'rb'))
    captions = encoder.encode(tmp_tags)    
    del encoder
    rest = len(images) % batch_size
    
    tags = tmp_tags
    tmp_images = images
    tmp_cap = captions.tolist()
    
    if rest != 0:
        for i in range(batch_size-rest):            
            tmp_images.append(images[i])
            tmp_cap.append(captions[i])
            tmp_tags.append(tags[i])

    print ('len: ', len(tmp_images), len(tmp_cap), len(tmp_tags))
    return tmp_images, tmp_cap, tmp_tags
    

num_epochs = 2400
learning_rate = 0.0001
resume_model = False
start_epoch = 0
image_dir = 'data/faces/'
trim_tags = 'data/tags.txt'
dis_updates = 1
gen_updates = 2
model_path = 'dcgan-model/'

params = dict(
    z_dim = 250,
    t_dim = 256,
    batch_size = 64,
    image_size = 64,
    gf_dim = 64,
    df_dim = 64,
    gfc_dim = 1024,
    dfc_dim = 1024,
    istrain = True,
    caption_length = 2400
)

# config = tf.ConfigProto(intra_op_parallelism_threads=8,
#                         inter_op_parallelism_threads=8)
#run_config = tf.ConfigProto()
#run_config.gpu_options.allow_growth = True

sess = tf.InteractiveSession()

gan = DCGAN.GAN(params)
input_tensors, variables, outputs, loss = gan.build_model()

images, captions, tags = read_images(image_dir , trim_tags , params['batch_size'])
#pickle.dump(captions , open('cap_b.pkl' , 'wb'))

#captions = pickle.load(open('cap_b.pkl' , 'rb'))



#d_optim = tf.train.AdamOptimizer(learning_rate, beta1 = 0.5).minimize(loss['d_loss'], var_list=variables['d_vars'])
#g_optim = tf.train.AdamOptimizer(learning_rate, beta1 = 0.5).minimize(loss['g_loss'], var_list=variables['g_vars'])


d_optim = tf.train.RMSPropOptimizer(learning_rate).minimize(loss['d_loss'], var_list=variables['d_vars'])
g_optim = tf.train.RMSPropOptimizer(learning_rate).minimize(loss['g_loss'], var_list=variables['g_vars'])

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(max_to_keep=100)

if resume_model:
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt != None:
        print ('path: ', ckpt.model_checkpoint_path)
        saver.restore(sess , ckpt.model_checkpoint_path)
        start_epoch = int(ckpt.model_checkpoint_path.split('-')[2])

for i in range(start_epoch, start_epoch + num_epochs+1):
    batch_no = 0
    while batch_no*params['batch_size'] < len(images):
        real_images, wrong_images, caption_vectors, z_noise, wrong_captions =\
            get_training_batch(batch_no, params['batch_size'], params['image_size'],
            params['z_dim'], params['caption_length'], images , captions)

        #dis update
        for j in range(dis_updates):
            #check_ts = [checks['d_loss1'] , checks['d_loss2'], checks['d_loss3']]
            _, d_loss, gen =\
            sess.run([d_optim, loss['d_loss'], outputs['generator']] ,
                    feed_dict = {
                        input_tensors['t_real_image'] : real_images,
                        input_tensors['t_wrong_image'] : wrong_images,
                        input_tensors['t_real_caption'] : caption_vectors,
                        input_tensors['t_z'] : z_noise,
                        input_tensors['t_wrong_caption'] : wrong_captions,
                    })

        

        # GEN UPDATE
        for j in range(gen_updates):
            _, g_loss, gen =\
            sess.run([g_optim, loss['g_loss'], outputs['generator']],
                    feed_dict = {
                        input_tensors['t_real_image'] : real_images,
                        input_tensors['t_wrong_image'] : wrong_images,
                        input_tensors['t_real_caption'] : caption_vectors,
                        input_tensors['t_z'] : z_noise,
                        input_tensors['t_wrong_caption'] : wrong_captions,
                    })
        #pdb.set_trace()
        print('d_loss = {:5f} g_loss = {:5f} batch_no = {} '
            'epochs = {}'.format(d_loss, g_loss, batch_no, i))
        print('-'*60)
        batch_no += 1
        
    if i%10 == 0:
        saver.save(sess, model_path + 'model', global_step=i)      

#images , captions = read_images(image_dir , trim_tags)

#test(image_dir , trim_tags)
#real_images, wrong_images, caption_vectors, z_noise = get_training_batch(0, 10, 96, 100, 2400, images , captions) 
#pdb.set_trace()
