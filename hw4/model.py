from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np

from skimage.transform import resize
from skimage.io import imread

from skip_thoughts import configuration
from skip_thoughts import encoder_manager

from ops import *
from utils import *
import scipy.misc
# from Utils import ops

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
    def __init__(self, sess, input_height=64, input_width=64, crop=False,
                 batch_size=64, sample_num = 64, output_height=64, output_width=64,
                 y_dim=2400, ry_dim=256, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='anime',
                 input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None, data_dir=None):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.y_dim = y_dim
        self.ry_dim = ry_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.data_dir = data_dir

        self.g_update = 2
        self.d_update = 1

        # batch normalization : deals with poor initialization helps gradient flow
        """
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        """
        """
        if self.dataset_name == 'mnist':
            self.data_X, self.data_y = self.load_mnist()
            self.c_dim = self.data_X[0].shape[-1]
        else:
            self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
            imreadImg = imread(self.data[0])
            if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
                self.c_dim = imread(self.data[0]).shape[-1]
            else:
                self.c_dim = 1
        """

        self.c_dim = 3
        self.grayscale = (self.c_dim == 1)

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')

        self.build_model()

    def build_model(self):
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')

        self.fake_inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='wrong_images')

        inputs = self.inputs

        self.z = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        self.G                  = self.generator(self.z, self.y)
        self.D, self.D_logits   = self.discriminator(inputs, self.y, reuse=False)
        self.sampler            = self.sampler(self.z, self.y)
        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
        self.D2_, self.D_logits2_ = self.discriminator(self.fake_inputs, self.y, reuse=True)
        
        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", self.G)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.d_loss_wrong = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits2_, tf.zeros_like(self.D2_)))
        self.g_loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                                                    
        self.d_loss = self.d_loss_real + self.d_loss_fake + self.d_loss_wrong

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):

        if self.dataset_name == 'anime':
            self.images, self.tags = self.load_anime()

        # d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
        #                     .minimize(self.d_loss, var_list=self.d_vars)
        # g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
        #                     .minimize(self.g_loss, var_list=self.g_vars)
        d_optim = tf.train.AdamOptimizer(config.learning_rate).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.g_sum = merge_summary([self.z_sum, self.d__sum,
            self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary(
                [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))

        if config.dataset == 'anime':
            sample_inputs = self.images[:self.sample_num]
            sample_labels = self.tags[:self.sample_num]
        
        """
        if config.dataset == 'mnist':
            sample_inputs = self.data_X[0:self.sample_num]
            sample_labels = self.data_y[0:self.sample_num]
        else:
            sample_files = self.data[0:self.sample_num]
            sample = [
                    get_image(sample_file,
                                        input_height=self.input_height,
                                        input_width=self.input_width,
                                        resize_height=self.output_height,
                                        resize_width=self.output_width,
                                        crop=self.crop,
                                        grayscale=self.grayscale) for sample_file in sample_files]
            if (self.grayscale):
                sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                sample_inputs = np.array(sample).astype(np.float32)
        """
    
        counter = 1
        start_time = time.time()
        could_load, checkpoint_epoch = self.load(self.checkpoint_dir)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in range(checkpoint_epoch, config.epoch + checkpoint_epoch):
            if config.dataset == 'anime':
                batch_idxs = min(len(self.images), config.train_size) // config.batch_size

            random_order = np.arange(len(self.images))
            np.random.shuffle(random_order)

            random_order2 = np.arange(len(self.images))
            for i in range(len(self.images)):
                while self.orig_tags[random_order[i]] == self.orig_tags[random_order2[i]]:
                    random_order2[i] = random.randint(0, len(self.images)-1)
            """
            while True:
                print ('Shuffling')
                np.random.shuffle(random_order2)
                allDiff = True
                for i in range(len(random_order2)):
                    i1 = random_order[i]
                    i2 = random_order2[i]
                    if self.orig_tags[i1] == self.orig_tags[i2]:
                        allDiff = False
                        for j in range (i+1, len(random_order2)):
                            if self.orig_tags[i1] != self.orig_tags[random_order2[j]]:
                                allDiff = True
                                random_order2[[i, j]] = random_order2[[j, i]]
                                break

                        if allDiff == False:
                            break
                        else:
                            i2 = random_order2[i]
                            if self.orig_tags[i1] == self.orig_tags[i2]:
                                print ('Wrong Swap')
                                allDiff = False
                                break

                if allDiff:
                    break
            """

            for i in range(len(random_order)):
                if self.orig_tags[random_order[i]] == self.orig_tags[random_order2[i]]:
                    errorF = open('error.log', 'a')
                    errorF.write('Error: ' + str(i) + ', ' + self.orig_tags[random_order[i]])
                    errorF.close()
                    break
            
            for idx in range(0, batch_idxs):
                if config.dataset == 'anime':
                    indices = random_order[idx*config.batch_size:(idx+1)*config.batch_size]
                    indices2 = random_order2[idx*config.batch_size:(idx+1)*config.batch_size]
                    batch_images = self.images[indices]
                    batch_labels = self.tags[indices]
                    batch_images2 = self.images[indices2]
                    
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                            .astype(np.float32)

                if config.dataset == 'anime':
                    # Update D network
                    for times in range(self.d_update):
                        _, summary_str = self.sess.run([d_optim, self.d_sum],
                            feed_dict={ 
                                self.inputs: batch_images,
                                self.z: batch_z,
                                self.y: batch_labels,
                                self.fake_inputs: batch_images2,
                            })
                        self.writer.add_summary(summary_str, counter)

                    # Update G network
                    for times in range(self.g_update):
                        _, summary_str = self.sess.run([g_optim, self.g_sum],
                            feed_dict={
                                self.z: batch_z, 
                                self.y:batch_labels,
                            })
                        self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    # _, summary_str = self.sess.run([g_optim, self.g_sum],
                    #     feed_dict={ self.z: batch_z, self.y:batch_labels })
                    self.writer.add_summary(summary_str, counter)
                    
                    errD_fake = self.d_loss_fake.eval({
                            self.z: batch_z, 
                            self.y:batch_labels
                    })
                    errD_real = self.d_loss_real.eval({
                            self.inputs: batch_images,
                            self.y:batch_labels
                    })
                    errG = self.g_loss.eval({
                            self.z: batch_z,
                            self.y: batch_labels
                    })

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 100) == 1:
                    if config.dataset == 'anime':
                        samples = self.sess.run(
                            [self.sampler],
                            feed_dict={
                                    self.z: sample_z,
                                    self.inputs: sample_inputs,
                                    self.y:sample_labels,
                            }
                        )
                        samples = samples[0]
                        save_images(samples, image_manifold_size(samples.shape[0]),
                                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                        # print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 

                # if np.mod(counter, 500) == 2:
                #     self.save(config.checkpoint_dir, counter)
                if epoch % 20 == 0:
                    self.save(config.checkpoint_dir, epoch)

    def discriminator(self, image, y, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, name = 'd_h0_conv')) #32
            h1 = lrelu( self.d_bn1(conv2d(h0, self.df_dim*2, name = 'd_h1_conv'))) #16
            h2 = lrelu( self.d_bn2(conv2d(h1, self.df_dim*4, name = 'd_h2_conv'))) #8
            h3 = lrelu( self.d_bn3(conv2d(h2, self.df_dim*8, name = 'd_h3_conv'))) #4
            
            # ADD TEXT EMBEDDING TO THE NETWORK
            reduced_text_embeddings = lrelu(linear(y, self.ry_dim, 'd_embedding'))
            reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,1)
            reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,2)
            tiled_embeddings = tf.tile(reduced_text_embeddings, [1,4,4,1], name='tiled_embeddings')
            
            h3_concat = tf.concat( [h3, tiled_embeddings], 3, name='h3_concat')
            h3_new = lrelu( self.d_bn4(conv2d(h3_concat, self.df_dim*8, 1,1,1,1, name = 'd_h3_conv_new'))) #4
            
            h4 = linear(tf.reshape(h3_new, [self.batch_size, -1]), 1, 'd_h3_lin')
            
            return tf.nn.sigmoid(h4), h4

    def generator(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            s = self.output_height
            s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
            
            reduced_text_embedding = lrelu( linear(y, self.ry_dim, 'g_embedding') )
            z_concat = tf.concat([z, reduced_text_embedding], 1)
            z_ = linear(z_concat, self.gf_dim*8*s16*s16, 'g_h0_lin')
            h0 = tf.reshape(z_, [-1, s16, s16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0))
            
            h1 = deconv2d(h0, [self.batch_size, s8, s8, self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1))
            
            h2 = deconv2d(h1, [self.batch_size, s4, s4, self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2))
            
            h3 = deconv2d(h2, [self.batch_size, s2, s2, self.gf_dim*1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3))
            
            h4 = deconv2d(h3, [self.batch_size, s, s, 3], name='g_h4')
            
            return (tf.tanh(h4)/2. + 0.5)

    def sampler(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
		
            s = self.output_height
            s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
            
            reduced_text_embedding = lrelu( linear(y, self.ry_dim, 'g_embedding') )
            z_concat = tf.concat([z, reduced_text_embedding], 1)
            z_ = linear(z_concat, self.gf_dim*8*s16*s16, 'g_h0_lin')
            h0 = tf.reshape(z_, [-1, s16, s16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train = False))
            
            h1 = deconv2d(h0, [self.batch_size, s8, s8, self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train = False))
            
            h2 = deconv2d(h1, [self.batch_size, s4, s4, self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train = False))
            
            h3 = deconv2d(h2, [self.batch_size, s2, s2, self.gf_dim*1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train = False))
            
            h4 = deconv2d(h3, [self.batch_size, s, s, 3], name='g_h4')
		
            return (tf.tanh(h4)/2. + 0.5)

    def load_skip_thought(self):
        MODEL_PATH="skip_thoughts_uni_2017_02_02/"
        VOCAB_FILE = MODEL_PATH + "vocab.txt"
        EMBEDDING_MATRIX_FILE = MODEL_PATH + "embeddings.npy"
        CHECKPOINT_PATH = MODEL_PATH + "model.ckpt-501424"
        
        self.encoder = encoder_manager.EncoderManager()
        self.encoder.load_model(configuration.model_config(),
                    vocabulary_file=VOCAB_FILE,
                    embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                    checkpoint_path=CHECKPOINT_PATH)

    def del_skip_thought(self):
        del self.encoder

    def load_anime(self):
        print ('loading anime images & skip thoughts...')

        images = []
        tags = []

        self.load_skip_thought()

        f = open(os.path.join(self.data_dir, 'trim.txt'))
        while (True):
            line = f.readline()
            if (len(line) == 0):
                break

            line = line[:-1]

            splits = line.split(',')
            if (len(splits[1]) > 0):
                index = splits[0]
                tag = splits[1]

                image = imread(os.path.join(self.data_dir, 'faces', index + '.jpg'))
                image = resize(image, (self.output_height, self.output_width))
                images.append(image)

                # tags.append(self.encoder.encode([tag])[0])
                tags.append(tag)

        self.orig_tags = tags

        tags = self.encoder.encode(tags)

        self.del_skip_thought()
        images = np.array(images)
        
        print ('Finish loading')

        return images, tags


    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
                self.dataset_name, self.batch_size,
                self.output_height, self.output_width)
            
    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            epoch = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, epoch
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def produceImages(self, testing_text_fileName, repeat=5):
        # could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        # if could_load:
        #     counter = checkpoint_counter
        #     print(" [*] Load SUCCESS")
        # else:
        #     print(" [!] Load failed...")

        
        indices = []
        tags = []
        f = open(os.path.join(self.data_dir, testing_text_fileName))
        while (True):
            line = f.readline()
            if (len(line) == 0):
                break

            if line[-1] == '\n':
                line = line[:-1]

            splits = line.split(',')
            indices.append(int(splits[0]))
            tags.append(splits[1])

        self.load_skip_thought()
        tags = self.encoder.encode(tags)
        self.del_skip_thought()
        tags = np.repeat(tags, repeat, axis=0)
        
        batch_idxs = (len(tags) + self.batch_size - 1) // self.batch_size

        for idx in range(batch_idxs):
            start = idx*self.batch_size
            limit = min(len(tags), (idx+1)*self.batch_size)
            batch_labels = np.zeros([self.batch_size, self.y_dim])
            batch_labels[:limit - start] = tags[start : limit]
            

            sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
            samples = self.sess.run(
                [self.sampler],
                feed_dict={
                        self.z: sample_z,
                        self.y: batch_labels,
                }
            )
            samples = samples[0]
            for i in range(start, limit):
                testing_text_id = indices[i // repeat]
                sample_id = i % repeat + 1

                scipy.misc.imsave('./{}/sample_{}_{}.jpg'.format(self.sample_dir, testing_text_id, sample_id), samples[i - start])


