import gym
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf

from collections import deque, namedtuple

class state_processor():
    def __init__(self, input_shape, scope_name="state_processor"):
        self.scope_name = scope_name
        with tf.variable_scope(self.scope_name):
            self.input_state = tf.placeholder(shape=input_shape, dtype=tf.float32)
            self.output = tf.squeeze(tf.image.rgb_to_grayscale(self.input_state))

    def process(self, sess, input_state):
        return sess.run(self.output, feed_dict={self.input_state: input_state})

class QNet():
    def __init__(self, scope_name, input_shape, lr, num_actions=4):
        self.scope_name = scope_name
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.lr = lr
        self.history_size = input_shape[-1]
        with tf.variable_scope(scope_name):
            self.build_model()

    def build_model(self):
        self.X = tf.placeholder(shape=[None] + list(self.input_shape), dtype=tf.float32, name="X")
        self.y = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        bs = tf.shape(self.X)[0]
        X_normal = self.X / 255
        self.conv1 = tf.contrib.layers.conv2d(self.X, 32, 5, 1, activation_fn=tf.nn.relu)
        self.pool1 = tf.contrib.layers.max_pool2d(self.conv1, 2, padding='SAME')
        self.conv2 = tf.contrib.layers.conv2d(self.pool1, 64, 5, 1, activation_fn=tf.nn.relu)
        self.pool2 = tf.contrib.layers.max_pool2d(self.conv2, 2, padding='SAME')
        self.conv3 = tf.contrib.layers.conv2d(self.pool2, 128, 5, 1, activation_fn=tf.nn.relu)
        self.pool3 = tf.contrib.layers.max_pool2d(self.conv3, 2, padding='SAME')
        self.conv4 = tf.contrib.layers.conv2d(self.pool3, 128, 5, 1, activation_fn=tf.nn.relu)
        self.pool4 = tf.contrib.layers.max_pool2d(self.conv4, 2, padding='SAME')
        self.conv5 = tf.contrib.layers.conv2d(self.pool4, 32, 5, 1, activation_fn=tf.nn.relu)

        self.flattened = tf.contrib.layers.flatten(self.conv5)
        self.fc1 = tf.contrib.layers.fully_connected(self.flattened, 512, activation_fn=tf.nn.relu)
        self.preds = tf.contrib.layers.fully_connected(self.fc1, self.num_actions)

        self.indices = tf.stack([tf.range(bs), self.actions], axis=1)
        self.action_preds = tf.gather_nd(self.preds, self.indices)

        self.loss = tf.reduce_mean(tf.squared_difference(self.y, self.action_preds))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_name)
        self.update_op = self.optimizer.minimize(self.loss, var_list=self.model_vars)

    def predict(self, sess, state):
        return sess.run(self.preds, feed_dict={self.X: state})

    def update(self, sess, state, actions, y):
        loss, _ = sess.run([self.loss, self.update_op], feed_dict={self.X: state,
                                                                   self.y: y,
                                                                   self.actions: actions})

def main():
    env = gym.make("Breakout-v0")
    a = QNet(input_shape=[210, 160, 1], scope_name="QNet", lr=1e-3)
    s = np.random.randn(2, 210, 160, 1)

if __name__ == '__main__':
    main()
