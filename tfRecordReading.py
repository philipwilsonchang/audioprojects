# Playing with tfRecord files
# Based off of: https://www.skcript.com/svr/why-every-tensorflow-developer-should-know-about-tfrecord/
# and: https://www.tensorflow.org/programmers_guide/datasets

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# Import training data
filenames = ["../nsynth-test.tfrecord"]
train_dataset = tf.data.TFRecordDataset(filenames)
print(train_dataset.output_classes)
print(train_dataset.output_types)
print(train_dataset.output_shapes)