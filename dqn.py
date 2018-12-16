import gym
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf

from collections import deque, namedtuple

class state_processor():
    def __init__(self, input_size, scope_name="state_processor"):
        self.scope_name = scope_name
        with tf.variable_scope(self.scope_name):
            self.input_state = tf.placeholder(shape=input_size, dtype=tf.float32)
            self.output = tf.squeeze(tf.image.rgb_to_grayscale(self.input_state))

    def process(self, sess, input_state):
        return sess.run(self.output, feed_dict={self.input_state: input_state})

class QNet():
    def __init__(self, scope_name, input_shape):
        self.scope_name = scope_name
        self.input_shape = input_shape
        self.history_size = input_shape[-1]
        with tf.variable_scope(scope):
            self.build_model()

    def build_model(self):
        self.X = tf.placeholder(shape=[None] + list(self.inpu_shape), dtype=tf.float32, name="X")
        self.y = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

def main():
    env = gym.make("Breakout-v0")
    a = state_processor(input_size=[210, 160, 3])
    s = np.random.randn(210, 160, 3)
    sess = tf.InteractiveSession()
    state = a.process(sess, s)
    print(state.shape)
    print(s.shape)
    print(s)
    print(sess)

if __name__ == '__main__':
    main()
