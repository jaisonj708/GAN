import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose
from preprocess import load_image_batch
import tensorflow_gan as tfgan
import tensorflow_hub as hub

import numpy as np

from imageio import imwrite
import os
import argparse

x  = tf.ones(shape=(4,5))
print(x)
print(x-0.2)
