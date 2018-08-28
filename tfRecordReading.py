# Playing with tfRecord files
# Based off of: https://www.skcript.com/svr/why-every-tensorflow-developer-should-know-about-tfrecord/
# and: https://www.tensorflow.org/programmers_guide/datasets

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# List all data in tfrecord (literally all data)
# for example in tf.python_io.tf_record_iterator("../nsynth-test.tfrecord"):
# 	print(tf.train.Example.FromString(example))


# Transforms tf.Dataset Example proto into an array of features containing data
# Input: TFRecord Example proto
# Output: Array of Tensors - one for each feature
def parse_tfrecord(tfrecord_example):
	feature_list = {'note': tf.FixedLenFeature(shape=[], dtype=tf.int64),
				'note_str': tf.FixedLenFeature(shape=[], dtype=tf.string),
				'instrument': tf.FixedLenFeature(shape=[], dtype=tf.int64),
				'instrument_str': tf.FixedLenFeature(shape=[], dtype=tf.string),
				'pitch': tf.FixedLenFeature(shape=[], dtype=tf.int64),
				'velocity': tf.FixedLenFeature(shape=[], dtype=tf.int64),
				'sample_rate': tf.FixedLenFeature(shape=[], dtype=tf.int64),
				'audio': tf.FixedLenFeature(shape=[1], dtype=tf.float64),
				'qualities': tf.FixedLenFeature(shape=[1], dtype=tf.int64),
				'qualities_str': tf.FixedLenFeature(shape=[1], dtype=tf.string),
				'instrument_family': tf.FixedLenFeature(shape=[], dtype=tf.int64),
				'instrument_family_str': tf.FixedLenFeature(shape=[], dtype=tf.string),
				'instrument_source': tf.FixedLenFeature(shape=[], dtype=tf.int64),
				'instrument_source_str': tf.FixedLenFeature(shape=[], dtype=tf.string)}
	parsed_features = tf.parse_single_example(tfrecord_example, feature_list)
	return parsed_features

# Import training data, read all examples, extract features
filenames = ["../nsynth-test.tfrecord"]
train_dataset = tf.data.TFRecordDataset(filenames)
train_dataset = train_dataset.map(parse_tfrecord)