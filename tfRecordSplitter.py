# Splits single TFRecord file into multiple TFRecord files, by count

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys

# Initialize TFRecord file writers
filename = "E:/NSynth/nsynth-train-0.tfrecord"
writer0 = tf.python_io.TFRecordWriter(filename)
filename = "E:/NSynth/nsynth-train-1.tfrecord"
writer1 = tf.python_io.TFRecordWriter(filename)
filename = "E:/NSynth/nsynth-train-2.tfrecord"
writer2 = tf.python_io.TFRecordWriter(filename)
filename = "E:/NSynth/nsynth-train-3.tfrecord"
writer3 = tf.python_io.TFRecordWriter(filename)
filename = "E:/NSynth/nsynth-train-4.tfrecord"
writer4 = tf.python_io.TFRecordWriter(filename)
filename = "E:/NSynth/nsynth-train-5.tfrecord"
writer5 = tf.python_io.TFRecordWriter(filename)
filename = "E:/NSynth/nsynth-train-6.tfrecord"
writer6 = tf.python_io.TFRecordWriter(filename)
filename = "E:/NSynth/nsynth-train-7.tfrecord"
writer7 = tf.python_io.TFRecordWriter(filename)
filename = "E:/NSynth/nsynth-train-8.tfrecord"
writer8 = tf.python_io.TFRecordWriter(filename)
filename = "E:/NSynth/nsynth-train-9.tfrecord"
writer9 = tf.python_io.TFRecordWriter(filename)
filename = "E:/NSynth/nsynth-train-10.tfrecord"
writer10 = tf.python_io.TFRecordWriter(filename)

# Set counter
count = 0

# Read examples from TFRecord, then write into other TFRecords
# Exact count splits obtained from JSON version of TFRecord
for example in tf.python_io.tf_record_iterator("E:/NSynth/nsynth-train.tfrecord"):
	if count < 65474:
		writer0.write(example)
	elif count < 78149:
		writer1.write(example)
	elif count < 86922:
		writer2.write(example)
	elif count < 119612:
		writer3.write(example)
	elif count < 171433:
		writer4.write(example)
	elif count < 205634:
		writer5.write(example)
	elif count < 240111:
		writer6.write(example)
	elif count < 254022:
		writer7.write(example)
	elif count < 273496:
		writer8.write(example)
	elif count < 278997:
		writer9.write(example)
	else:
		writer10.write(example)

	if count % 1000 == 0:
		print("Examples processed: ", count)

	count += 1


writer0.close()
writer1.close()
writer2.close()
writer3.close()
writer4.close()
writer5.close()
writer6.close()
writer7.close()
writer8.close()
writer9.close()
writer10.close()
sys.stdout.flush()