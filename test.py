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


# THIS IS A TESTNG GROUND ONLY


# print(tf.keras.losses.binary_crossentropy(y_pred=tf.constant([[0.5],[0.6],[0.7]]), y_true=tf.constant([[0],[0],[0]])))
# print(-tf.math.log(1-0.6))
class Discriminator_Model(tf.keras.Model):
    def __init__(self):
        super(Discriminator_Model, self).__init__()
        """
        The model for the discriminator network is defined here.
        """

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.channels = [3,64,128,256,512]

        self.model = tf.keras.Sequential()
        m = self.model

        m.add(Conv2D(filters=self.channels[1], kernel_size=(5,5), strides=(2,2), padding='same', input_shape=(64,64,self.channels[0])))
        m.add(LeakyReLU())
        for num in self.channels[2:]:
            m.add(Conv2D(filters=num, kernel_size=(5,5), strides=(2,2), padding='same'))
            m.add(LeakyReLU())
            m.add(BatchNormalization(axis=-1))
        m.add(Flatten())
        m.add(Dense(1, activation='sigmoid'))
        m.compile(optimizer=self.optimizer,loss=tf.keras.losses.binary_crossentropy) # for graph building purposes

    @tf.function
    def call(self, inputs):
        """
        Executes the discriminator model on a batch of input images and outputs whether it is real or fake.

        :param inputs: a batch of images, shape=[batch_size, height, width, channels]

        :return: a batch of values indicating whether the image is real or fake, shape=[batch_size, 1]
        """
        return self.model.predict_on_batch(inputs)


class Generator_Model(tf.keras.Model):
    def __init__(self):
        """
        The model for the generator network is defined here.
        """
        super(Generator_Model, self).__init__()
        # TODO: Define the model, loss, and optimizer

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.channels = [512,256,128,64,3]

        self.model = tf.keras.Sequential()
        m = self.model

        m.add(Dense(4*4*512, input_shape=(100,), activation='relu'))
        m.add(Reshape((4,4,512)))
        for num in self.channels[1:-1]:
            m.add(Conv2DTranspose(filters=num, kernel_size=(5,5), strides=(2,2), padding='same', activation='relu'))
            m.add(BatchNormalization(axis=-1))
        num = self.channels[-1]
        m.add(Conv2DTranspose(filters=num, kernel_size=(5,5), strides=(2,2), padding='same', activation=tf.keras.activations.tanh))
        m.compile(optimizer=self.optimizer,loss=tf.keras.losses.binary_crossentropy)

    def call(self, inputs):
        return self.model.predict_on_batch(inputs)

    @tf.function
    def loss_function(self, disc_fake_output):
        """
        Outputs the loss given the discriminator output on the generated images.

        :param disc_fake_output: the discriminator output on the generated images, shape=[batch_size,1]

        :return: loss, the cross entropy loss, scalar
        """
        faked = disc_fake_output
        losses = tf.keras.losses.binary_crossentropy(y_pred=faked, y_true=tf.ones_like(faked))
        return tf.reduce_mean(losses)


generator = Generator_Model()
discriminator = Discriminator_Model()

with tf.GradientTape(persistent=True) as tape:
    noise = tf.random.uniform(shape=[128,100], minval=-1, maxval=1)
    fake_imgs = generator(noise)

    fake_pred = discriminator(fake_imgs)
    # real_pred = discriminator(batch)

    g_loss = generator.loss_function(fake_pred)
    # d_loss = discriminator.loss_function(real_pred,fake_pred)

grads = tape.gradient(g_loss, generator.model.trainable_variables)
generator.optimizer.apply_gradients(zip(grads, generator.model.trainable_variables))
# grads = tape.gradient(g_loss, generator.model.weights) # is this necessary ??
# generator.optimizer.apply_gradients(zip(grads, generator.model.weights))

# grads = tape.gradient(d_loss, discriminator.model.weights)
# discriminator.optimizer.apply_gradients(zip(grads, discriminator.model.weights))





# mod = Generator_Model()
# print(mod.m.layers)
# print(mod.m.weights)
# #
# x = tf.ones(shape=(128,100))
# print(mod.m.predict_on_batch(x))
