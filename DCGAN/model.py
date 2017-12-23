from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import scipy.spatial.distance as sd
import numpy as np
from six.moves import xrange
from pycocotools.coco import COCO
import scipy.spatial.distance as sd
from sklearn.decomposition import PCA

from ops import *
from utils import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
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
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bne = batch_norm(name='d_bne')

    #if not self.y_dim:
    self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')
    self.g_bne = batch_norm(name='g_bne')
    #if not self.y_dim:
    self.g_bn3 = batch_norm(name='g_bn3')

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir

    if 'cond' in self.dataset_name:
        #self.data = np.load('bird_mappings.npy')
#        self.data = np.load('new_mapping.npy')
#        x = np.array([x[0] for x in self.data])
#        compressor = PCA(n_components=128)
#        compressor.fit(x)
#        x_ = compressor.transform(x)
#        self.data = np.array([(x_[i], self.data[i][1]) for i in range(len(self.data))])
        self.data = np.load('new_new_mapping.npy')
        #self.data = np.load('bird_mappings_big.npy')
        #self.data = np.array([(x[0][0:100],x[1]) for x in self.data])
        #self.data_X, self.data_y = [x[1] for x in self.data],[y[0] for y in self.data]#self.load_mnist()
        self.c_dim = 3
    else:
      #self.data = glob(os.path.join("../celebA/img_align_celeba/", self.input_fname_pattern))
      #self.data = np.load('new_mapping.npy')
      #self.data = np.load('bird_mappings.npy')#glob(os.path.join("../coco/train2014/", self.input_fname_pattern))
      #glob(os.path.join("../celebA/img_align_celeba/", self.input_fname_pattern))
      self.data = np.array([x[1] for x in self.data])
      #imreadImg = imread(self.data[0])
      #if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
      #  self.c_dim = imread(self.data[0]).shape[-1]
      #else:
      #  self.c_dim = 1
      #scores = sd.cdist([self.data[0][0]],[x[0] for x in self.data], 'cosine')[0]
      #sidx = np.argsort(scores)
      #self.data = self.data[sidx[:10000]]
      self.c_dim = 3

    self.grayscale = (self.c_dim == 1)

    self.build_model()

  def build_model(self):
    if self.y_dim:
      self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
      self.y_shuffle = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y_shuffle')
    else:
      self.y = None

    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')

    inputs = self.inputs

    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

    self.G                  = self.generator(self.z, self.y)
    self.D, self.D_logits   = self.discriminator(inputs, self.y, reuse=False)
    self.sampler            = self.sampler(self.z, self.y)
    self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
    self.D_shuffle, self.D_logits_shuffle = self.discriminator(inputs, self.y_shuffle, reuse=True)
    
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
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))  + tf.reduce_mean(
              sigmoid_cross_entropy_with_logits(self.D_logits_shuffle, tf.zeros_like(self.D_shuffle)))
    self.g_loss = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                          
    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()
    
  def train(self, config):
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
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
    
#    print('Preprocessing CUB data...')
#    self.data = [(x[0], get_image(x[1],
#                    input_height=self.input_height,
#                    input_width=self.input_width,
#                    resize_height=self.output_height,
#                    resize_width=self.output_width,
#                    crop=self.crop,
#                    grayscale=self.grayscale,
#                    coco=None)) for x in self.data]
    coco = COCO('../coco/annotations/captions_train2014.json')
    print('Preprocessing COCO data...')
    if 'cond' in self.dataset_name:
        imgIds = [x[1] for x in self.data]
    else:
        imgIds = self.data
    caps = coco.loadAnns(ids=coco.getAnnIds(imgIds=imgIds))
    true_img_ids = []
    for cap in caps:
        lst = cap['caption'].split(' ')
        if 'car' in lst or 'truck' in lst:
            true_img_ids.append(cap['image_id'])
    
    true_img_ids = np.array(true_img_ids)
    true_img_ids = np.unique(true_img_ids)
    all_imgs = {img_id: get_image(img_id,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale,
                    coco=coco) for img_id in true_img_ids}
    #print('Saving Stuff, brace yourself..')
    #np.save('vehicle_imgs.npy', all_imgs)
    #print('Saved the stuff, exit now dude!.....')
    #time.sleep(10)
    true_img_ids = {x:1 for x in true_img_ids}
    if 'cond' in self.dataset_name:
        self.data = self.data[[(self.data[i][1] in true_img_ids) for i in range(len(self.data))]]
        self.data = np.array([(self.data[i][0], all_imgs[self.data[i][1]]) for i in range(len(self.data))])
    else:
        self.data = self.data[[(self.data[i] in true_img_ids) for i in range(len(self.data))]]
    print('Preprocessing Complete. Testing ...')
#    if 'cond' in self.dataset_name:  
#        imgIds = [x[1] for x in self.data]
#        imgEnc = [x[0] for x in self.data]
#        for i in range(5):
#            idx = np.random.randint(0, len(imgIds))
#            dists = sd.cdist([imgEnc[idx]], imgEnc, 'cosine')[0]
#            sorted_ids = np.argsort(dists)
#            print('Closest to - ', imgIds[idx])
#            for j in range(10):
#                print(imgIds[sorted_ids[j]])
    
    #catIdList = coco.getCatIds(catNms=['car', 'bus', 'airplane', 'truck'])
    #imgIds = []
    #for i in range(len(catIdList)):
    #    imgIds.extend(coco.getImgIds(catIds=catIdList[i]))
    #imgIds = {x:1 for x in imgIds}
    #self.data = self.data[[(self.data[i][1] in imgIds) for i in range(len(self.data))]]
#    for i in range(len(self.data)):
#        print('An image id - ', self.data[i][1])
    if 'cond' in config.dataset:
        self.data_X, self.data_y = np.array([x[1] for x in self.data]),np.array([y[0] for y in self.data],dtype=np.float32)
    
    if 'cond' in config.dataset:
      sample_idx = np.random.randint(0, self.data_X.shape[0], [self.sample_num])  
      #sample_files = self.data_X[sample_idx]
      sample_labels = self.data_y[sample_idx]
      #sample_sentences = [x[2] for x in self.data[sample_idx]]
      sample_inputs = self.data_X[sample_idx]
      sample_inputs = np.array(sample_inputs).astype(np.float32)
    else:
      sample_files = self.data[0:self.sample_num]
      sample = [
          get_image(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale,
                    coco=coco) for sample_file in sample_files]
      if (self.grayscale):
        sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
      else:
        sample_inputs = np.array(sample).astype(np.float32)
  
    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in xrange(config.epoch):
      if 'cond' in config.dataset:
        batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
      else:      
        #self.data = #glob(os.path.join(
          #"../celebA/img_align_celeba/", self.input_fname_pattern))
          #"../coco/train2014/", self.input_fname_pattern))
        batch_idxs = min(len(self.data), config.train_size) // config.batch_size

      for idx in xrange(0, batch_idxs):
        if 'cond' in config.dataset:
          cur_batch = np.random.randint(0, self.data_X.shape[0], [config.batch_size]) 
          shuffle_batch = np.random.randint(0, self.data_X.shape[0], [config.batch_size])
          #batch_files = self.data_X[cur_batch]
          batch_labels = self.data_y[cur_batch]
          batch_shuffle = self.data_y[shuffle_batch]
          batch_images = self.data_X[cur_batch]
          batch_images = np.array(batch_images).astype(np.float32)
        else:
          batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
          batch = [
              get_image(batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        crop=self.crop,
                        grayscale=self.grayscale,
                        coco=coco) for batch_file in batch_files]
          if self.grayscale:
            batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
          else:
            batch_images = np.array(batch).astype(np.float32)

        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
              .astype(np.float32)

        if 'cond' in config.dataset:
          # Update D network
          _, summary_str = self.sess.run([d_optim, self.d_sum],
            feed_dict={ 
              self.inputs: batch_images,
              self.z: batch_z,
              self.y: batch_labels,
              self.y_shuffle: batch_shuffle
            })
          self.writer.add_summary(summary_str, counter)

          # Update G network
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={
                    self.inputs: batch_images,
              self.z: batch_z, 
              self.y: batch_labels,
              self.y_shuffle: batch_shuffle
            })
          self.writer.add_summary(summary_str, counter)

          # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.inputs: batch_images, self.z: batch_z, self.y:batch_labels, self.y_shuffle: batch_shuffle })
          self.writer.add_summary(summary_str, counter)
          
#          _, summary_str = self.sess.run([g_optim, self.g_sum],
#            feed_dict={ self.inputs: batch_images, self.z: batch_z, self.y:batch_labels, self.y_shuffle: batch_shuffle })
#          self.writer.add_summary(summary_str, counter)
          
          errD_fake = self.d_loss_fake.eval({
              self.inputs: batch_images,    
              self.z: batch_z, 
              self.y:batch_labels,
              self.y_shuffle: batch_shuffle
          })
          errD_real = self.d_loss_real.eval({
              self.inputs: batch_images,
              self.y: batch_labels,
              self.y_shuffle: batch_shuffle
          })
          errG = self.g_loss.eval({
              self.z: batch_z,
              self.y: batch_labels,
              self.y_shuffle: batch_shuffle
          })
        else:
          # Update D network
          _, summary_str = self.sess.run([d_optim, self.d_sum],
            feed_dict={ self.inputs: batch_images, self.z: batch_z })
          self.writer.add_summary(summary_str, counter)

          # Update G network
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.z: batch_z })
          self.writer.add_summary(summary_str, counter)

          # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.z: batch_z })
          self.writer.add_summary(summary_str, counter)
          
          # Thrice babyyyy
#          _, summary_str = self.sess.run([g_optim, self.g_sum],
#            feed_dict={ self.z: batch_z })
#          self.writer.add_summary(summary_str, counter)
          
          errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
          errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
          errG = self.g_loss.eval({self.z: batch_z})

        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG))

        if np.mod(counter, 50) == 1:
          if 'cond' in config.dataset:
            sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
            samples, d_loss, g_loss = self.sess.run(
              [self.sampler, self.d_loss, self.g_loss],
              feed_dict={
                  self.z: sample_z,
                  self.inputs: sample_inputs,
                  self.y: sample_labels,
                  self.y_shuffle: batch_shuffle,
              }
            )
            save_images(samples, image_manifold_size(samples.shape[0]),
                  './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
            #print(sample_files)
          else:
            try:
              samples, d_loss, g_loss = self.sess.run(
                [self.sampler, self.d_loss, self.g_loss],
                feed_dict={
                    self.z: sample_z,
                    self.inputs: sample_inputs,
                },
              )
              save_images(samples, image_manifold_size(samples.shape[0]),
                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
              print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
            except:
              print("one pic error!...")

        if np.mod(counter, 200) == 2:
          self.save(config.checkpoint_dir, counter)

  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      if not self.y_dim:
        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

        return tf.nn.sigmoid(h4), h4
      else:
#        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
#        x = image
#        #x = conv_cond_concat(image, yb)
#
#        h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
#        #h0 = conv_cond_concat(h0, yb)
#
#        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
#        h1 = tf.reshape(h1, [self.batch_size, -1])      
#        h1 = concat([h1, y], 1)
#        
#        h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
#        #h2 = concat([h2, y], 1)
#
#        h3 = linear(h2, 1, 'd_h3_lin')
#        
#        return tf.nn.sigmoid(h3), h3
#        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
#        x = image
#        x = conv_cond_concat(image, yb)
        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        #e1 = lrelu(self.d_bne(linear(y, self.df_dim*4, 'd_e1_lin')))
        #e1 = tf.reshape(e1, [self.batch_size, 1, 1, -1])
        yb = tf.reshape(y, [self.batch_size, 1, 1, -1])
        h1 = conv_cond_concat(h1, yb)
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        h3 = tf.reshape(h3, [self.batch_size, -1])
        #h3 = concat([h3, y], 1)
        h4 = linear(h3, 1, 'd_h4_lin')

        return tf.nn.sigmoid(h4), h4

  def generator(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        self.z_, self.h0_w, self.h0_b = linear(
            z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(
            self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = deconv2d(
            h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = deconv2d(
            h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv2d(
            h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv2d(
            h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)
      else:
#        s_h, s_w = self.output_height, self.output_width
#        s_h2, s_h4 = int(s_h/2), int(s_h/4)
#        s_w2, s_w4 = int(s_w/2), int(s_w/4)
#
#        # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
#        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
#        z = concat([z, y], 1)
#
#        h0 = tf.nn.relu(
#            self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
#        #h0 = concat([h0, y], 1)
#
#        h1 = tf.nn.relu(self.g_bn1(
#            linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
#        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
#
#        #h1 = conv_cond_concat(h1, yb)
#
#        h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
#            [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
#        #h2 = conv_cond_concat(h2, yb)
#
#        return tf.nn.sigmoid(
#            deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        #e1 = tf.nn.relu(self.g_bne(linear(y, self.df_dim*4, 'g_e1_lin')))
        z = concat([z, y], 1)
        self.z_, self.h0_w, self.h0_b = linear(
            z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(
            self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = deconv2d(
            h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = deconv2d(
            h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv2d(
            h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv2d(
            h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)

  def sampler(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        h0 = tf.reshape(
            linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
            [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))

        h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))

        h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))

        h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

        return tf.nn.tanh(h4)
      else:
#        s_h, s_w = self.output_height, self.output_width
#        s_h2, s_h4 = int(s_h/2), int(s_h/4)
#        s_w2, s_w4 = int(s_w/2), int(s_w/4)
#
#        # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
#        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
#        z = concat([z, y], 1)
#
#        h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
#        #h0 = concat([h0, y], 1)
#
#        h1 = tf.nn.relu(self.g_bn1(
#            linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
#        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
#        #h1 = conv_cond_concat(h1, yb)
#
#        h2 = tf.nn.relu(self.g_bn2(
#            deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
#        #h2 = conv_cond_concat(h2, yb)
#
#        return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        #e1 = tf.nn.relu(self.g_bne(linear(y, self.df_dim*4, 'g_e1_lin')))
        z = concat([z, y], 1)
        h0 = tf.reshape(
            linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
            [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))

        h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))

        h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))

        h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

        return tf.nn.tanh(h4)

  def load_mnist(self):
    data_dir = '../mnist/'#os.path.join("", self.dataset_name)
    
    fd = open(os.path.join(data_dir,'train-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)
    
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)
    
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i,y[i]] = 1.0
    
    return X/255.,y_vec

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
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'DCGAN.model-27202'))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0