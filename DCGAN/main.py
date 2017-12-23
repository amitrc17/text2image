import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json, show_all_variables
from sklearn.decomposition import PCA
from pycocotools.coco import COCO

import tensorflow as tf
from ops import *
from utils import *


reuse = tf.AUTO_REUSE

flags = tf.app.flags
flags.DEFINE_integer("epoch", 5000, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "cond_cls_pca_coco_vehicle", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples_cond_cls_pca_coco_vehicle", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")
FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  #device_count={
#          'GPU': 0
#          })
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
    if 'cond' in FLAGS.dataset:
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          sample_num=FLAGS.batch_size,
          y_dim=128,
          z_dim=FLAGS.generate_test_images,
          dataset_name=FLAGS.dataset,
          input_fname_pattern=FLAGS.input_fname_pattern,
          crop=FLAGS.crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir)
    else:
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          sample_num=FLAGS.batch_size,
          z_dim=FLAGS.generate_test_images,
          dataset_name=FLAGS.dataset,
          input_fname_pattern=FLAGS.input_fname_pattern,
          crop=FLAGS.crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir)

    show_all_variables()

    if FLAGS.train:
      dcgan.train(FLAGS)
    else:
      if not dcgan.load(FLAGS.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")
#      data = np.load('new_mapping.npy')
#      x = np.array([x[0] for x in data])
#      compressor = PCA(n_components=128)
#      compressor.fit(x)
#      x_ = compressor.transform(x)
#      data = np.array([(x_[i], data[i][1]) for i in range(len(data))])
#      np.save('new_new_mapping.npy', data)
      #data = np.load('new_new_mapping.npy') 
      #data = np.load('bird_mappings_big.npy')
      data = np.load('bird_mappings.npy')
#      coco = COCO('../coco/annotations/captions_train2014.json')
#      print('Preprocessing COCO data...')
#      imgIds = [x[1] for x in data]
#      caps = coco.loadAnns(ids=coco.getAnnIds(imgIds=imgIds))
#      true_img_ids = []
#      for cap in caps:
#          lst = cap['caption'].split(' ')
#          if 'car' in lst or 'truck' in lst:
#              true_img_ids.append(cap['image_id'])
#    
#      true_img_ids = np.array(true_img_ids)
#      true_img_ids = np.unique(true_img_ids)
##      all_imgs = {img_id: get_image(img_id,
##                    input_height=self.input_height,
##                    input_width=self.input_width,
##                    resize_height=self.output_height,
##                    resize_width=self.output_width,
##                    crop=self.crop,
##                    grayscale=self.grayscale,
##                    coco=coco) for img_id in true_img_ids}
#      #print('Saving Stuff, brace yourself..')
#      #np.save('vehicle_imgs.npy', all_imgs)
#      #print('Saved the stuff, exit now dude!.....')
#      #time.sleep(10)
#      true_img_ids = {x:1 for x in true_img_ids}
#      data = data[[(data[i][1] in true_img_ids) for i in range(len(data))]]
      #data = np.array([(data[i][0], [data[i][1]]) for i in range(len(data))])
      b_size = 64
      tot = 20000
#      birds_gt = [get_image(x[1],
#                    input_height=dcgan.input_height,
#                    input_width=dcgan.input_width,
#                    resize_height=dcgan.output_height,
#                    resize_width=dcgan.output_width,
#                    crop=dcgan.crop,
#                    grayscale=dcgan.grayscale,
#                    coco=None) for x in data]
      print(data.shape)
      ret = np.empty((0, 64, 64, 3), dtype=np.float32)
      #ret = np.array(birds_gt)
      data = np.array([(x[0], x[1][19:-4]) for x in data])
      for i in range(tot//b_size):
          test_idx = np.random.randint(0, len(data)) 
          print('Loading text embedding - ', data[test_idx][1])
          inp = np.random.uniform(-1.0,1.0, size=[b_size, 50])
          samples = sess.run(dcgan.sampler, feed_dict={
                  dcgan.z: inp,
                  dcgan.y: np.array([data[test_idx][0]]*b_size)
                  })
          print('Done with ', i, ' ...')
          print(samples.shape)
          save_images(samples, image_manifold_size(samples.shape[0]),
                  './{}/train_{}_{:04d}.png'.format('output_cond_cub', data[test_idx][1], 64))
          ret = np.vstack((ret, samples))
    print(ret.shape)
    ret = (ret+1.)/2
    np.save('imgs_cond_CUB_bird.npy', ret)
      

    # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
    #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
    #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
    #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
    #                 [dcgan.h4_w, dcgan.h4_b, None])

    # Below is codes for visualization
    #OPTION = 1
    #visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
  tf.app.run()
