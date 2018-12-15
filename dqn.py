import gym
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf

from collections import deque, namedtuple

class state_processor():
    def __init__(self, input_size):
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=input_size, dtype=tf.uint8)
            self.output = tf.squeeze(tf.image.rgb_to_grayscale(self.input_state))

	def process(self, sess, inpu_state):
        return sess.run(self.output, feed_dict={self.input_state: input_state})

def main():
    env = gym.make("Breakout-v0")

if __name__ == '__main__':
    main()
