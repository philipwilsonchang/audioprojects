# Playing with tfRecord files
# Based off of: https://www.skcript.com/svr/why-every-tensorflow-developer-should-know-about-tfrecord/
# and: https://www.tensorflow.org/programmers_guide/datasets

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from scipy import signal

SAMPLERATE = 16000

# List all data in tfrecord (literally all data)
# for example in tf.python_io.tf_record_iterator("../nsynth-test.tfrecord"):
# 	print(tf.train.Example.FromString(example))

feature_list = {'note': tf.FixedLenFeature(shape=[], dtype=tf.int64),
				'note_str': tf.FixedLenFeature(shape=[], dtype=tf.string),
				'instrument': tf.FixedLenFeature(shape=[], dtype=tf.int64),
				'instrument_str': tf.FixedLenFeature(shape=[], dtype=tf.string),
				'pitch': tf.FixedLenFeature(shape=[], dtype=tf.int64),
				'velocity': tf.FixedLenFeature(shape=[], dtype=tf.int64),
				'sample_rate': tf.FixedLenFeature(shape=[], dtype=tf.int64),
				'audio': tf.FixedLenFeature(shape=[1], dtype=tf.float32),
				'qualities': tf.FixedLenFeature(shape=[1], dtype=tf.int64),
				'qualities_str': tf.FixedLenFeature(shape=[1], dtype=tf.string),
				'instrument_family': tf.FixedLenFeature(shape=[], dtype=tf.int64),
				'instrument_family_str': tf.FixedLenFeature(shape=[], dtype=tf.string),
				'instrument_source': tf.FixedLenFeature(shape=[], dtype=tf.int64),
				'instrument_source_str': tf.FixedLenFeature(shape=[], dtype=tf.string)}

# Import training data, read all examples, extract features
# tfrecordname = ["../data/nsynth-test.tfrecord"]
# train_dataset = tf.data.TFRecordDataset(tfrecordname) # Parse entire dataset
# train_dataset = train_dataset.map(parse_tfrecord)

# Import single example of training data
filename_queue = tf.train.string_input_producer(["../data/nsynth-test.tfrecord"])
reader = tf.TFRecordReader()
_, serialized = reader.read(filename_queue)
features = tf.parse_single_example(serialized, feature_list)

# STFT
stft_out = tf.contrib.signal.stft(features["audio"], frame_length=SAMPLERATE * 4, frame_step=SAMPLERATE, fft_length=1024)
power_spectrograms = tf.real(stft_out * tf.conj(stft_out))
print(power_spectrograms)
