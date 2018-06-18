# From https://towardsdatascience.com/tensorflow-for-absolute-beginners-28c1544fb0d6 for exploration purposes


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

T, F = 1.0, -1.0   #True has the +1.0 value and False has -1.0, it's important to note that
# you can assign any value to them
bias = 1.0
training_input = [
    [T, T, bias],
    [T, F, bias],
    [F, T, bias],
    [F, F, bias],
]
training_output = [
    [T],
    [F],
    [F],
    [F],
]
w = tf.Variable(tf.random_normal([3, 1]), dtype=tf.float32)
# step(x) = { 1 if x > 0; -1 otherwise }
def step(x):
    is_greater = tf.greater(x, 0)
    as_float = tf.to_float(is_greater)
    doubled = tf.multiply(as_float, 2)
    return tf.subtract(doubled, 1)

output = step(tf.matmul(training_input, w))
error = tf.subtract(training_output, output)
mse = tf.reduce_mean(tf.square(error))

delta = tf.matmul(training_input, error, transpose_a=True)
train = tf.assign(w, tf.add(w, delta))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

err, target = 1, 0
epoch, max_epochs = 0, 10

while err > target and epoch < max_epochs:
    epoch += 1
    err, _ = sess.run([mse, train])
print('epoch:', epoch, 'mse:', err)

print(sess.run(w))