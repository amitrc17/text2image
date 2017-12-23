# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 08:41:35 2017

@author: amitrc
"""
import collections
import os
from tempfile import gettempdir
import zipfile

import numpy as np

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from skip_thoughts import configuration
from skip_thoughts import encoder_manager
import scipy.spatial.distance as sd
from pycocotools.coco import COCO
from sklearn.decomposition import PCA

# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'


# pylint: disable=redefined-outer-name
def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  local_filename = os.path.join(gettempdir(), filename)
  if not os.path.exists(local_filename):
    local_filename, _ = urllib.request.urlretrieve(url + filename,
                                                   local_filename)
  statinfo = os.stat(local_filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception('Failed to verify ' + local_filename +
                    '. Can you get to it with a browser?')
  return local_filename


filename = maybe_download('text8.zip', 31344016)


# Read the data into a list of strings.
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

vocabulary = read_data(filename)
print('Data size', len(vocabulary))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000


def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  print('Total distinct words:', len(dictionary))
  return data, count, dictionary, reversed_dictionary

# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
#data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
#                                                            vocabulary_size)
#del vocabulary  # Hint to reduce memory.
#print('Most common words (+UNK)', count[:5])
#print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
#
  
#cap_path = 'coco/annotations/captions_train2014.json'
#cat_path = 'coco/annotations/instances_train2014.json'
#print('Loading Captions')
#coco_cap = COCO(cap_path)
#captions = coco_cap.loadAnns(coco_cap.getAnnIds())
#words = [caption['caption'] for caption in captions]

encoder = encoder_manager.EncoderManager()
encoder.load_model(configuration.model_config(), vocabulary_file='checkpoints/skipthought/vocab.txt',
                   embedding_matrix_file='checkpoints/skipthought/embeddings.npy',
                   checkpoint_path='checkpoints/skipthought/model.ckpt-501424')

compressor = PCA(n_components=128)
print('Loading Captions')
bird_caps = np.load('bird_caps_big.npy')
print('loaded captions -', len(bird_caps))
words = [bird_cap[0] for bird_cap in bird_caps]
##words = [key for key in dictionary.keys()]
encodings = encoder.encode(words, verbose=True)
print('Performing PCA...')
compressor.fit(np.array(encodings))
encodings = compressor.transform(encodings)
mapping = []

for idx, encoding in enumerate(encodings):
    mapping.append((encoding, bird_caps[idx][1]))
mapping = np.array(mapping)
np.save('bird_mappings_big.npy', mapping)
#np.save('wordencodings.npy', mapping)
#mapping = np.load('new_mapping.npy')
#print(mapping[0:5])
#np.save('new_mapping.npy', new_mapping)
#print(mapping)
    
#for i in range(5):
#    idx = np.random.randint(0,len(words))
#    encoding = encodings[idx]
#    scores = sd.cdist([encoding],encodings, 'cosine')[0]
#    sorted_ids = np.argsort(scores)
#    print('SENTENCE = ', words[idx])
#    print('Closest sentences - ')
#    for j in range(10):
#        print(words[sorted_ids[j]])

# Step 6: Visualize the embeddings.


# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

#try:
#  # pylint: disable=g-import-not-at-top
#  from sklearn.manifold import TSNE
#  import matplotlib.pyplot as plt
#
#  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
#  plot_only = 500
#  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
#  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
#  plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))
#
#except ImportError as ex:
#  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
#  print(ex)

