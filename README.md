# Audio Projects
This project contains quick and dirty Python scripts to experiment with audio processing. Most scripts revolve around attempting to train a CNN in Tensorflow to recognize musical instruments in 4 second audio samples, taken from the [NSynth dataset](https://magenta.tensorflow.org/datasets/nsynth).

The NSynth dataset contains 4 second samples of different instruments playing different notes, categorizing instruments into 11 distinct families. The goal was to train a CNN using the spectrogram of each sample to identify which instrument family an audio sample uses.

# Requirements

These scripts were developed with Python 3.6.5 and Jupyter Notebook 4.4.0. Additionally, Numpy 1.14.5, and TensorFlow 1.10.0 was used.


# Running

The CNN model code is stored in cnn-audio-classifier.ipynb and can be run in an instance of Jupyter Notebook.

tfRecordSplitter.py splits the NSynth training tfrecord file into 11 uneven shards, split by approximate location of the divisions of instrument families in the original tfrecord file. This is to encourage more even shuffling for training, as the tfrecord is roughly ordered by instrument families. This causes batches to have the same target classification, causing the CNN weights to converge erroneously.

# Results

The model had a peak accuracy of 28% after Epoch 3, after which the validation accuracy began to drop. This performance is markedly better than random selection (9% expected accuracy) but is still low. This indicates that a lot of information was lost in converting the 4-second sample into a spectrogram. Future work would explore incorporating a time dimension into the model input.
