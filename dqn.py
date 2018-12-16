import gym
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
import csv

from collections import namedtuple

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
        return loss

class param_copier():
    def __init__(self, qnet, target_net):
        qnet_params = [t for t in tf.trainable_variables() if t.name.startswith(qnet.scope_name)]
        qnet_params = sorted(qnet_params, key=lambda v: v.name)
        target_net_params = [t for t in tf.trainable_variables() if t.name.startswith(target_net.scope_name)]
        target_net_params = sorted(target_net_params, key=lambda v: v.name)

        self.copy_ops = []
        for q_v, t_v in zip(qnet_params, target_net_params):
            cp = t_v.assign(q_v)
            self.copy_ops.append(cp)

    def copy(self, sess):
        sess.run(self.copy_ops)

def epsilon_greedy(qnet, num_actions):
    def policy(sess, state, epsilon):
        pol = np.ones(num_actions, dtype=float) * epsilon / num_actions
        q_values = qnet.predict(sess, np.expand_dims(state, 0))[0]
        greedy_action = np.argmax(q_values)
        pol[greedy_action] += (1.0 - epsilon)
        return pol
    return policy

def train(train_episodes, save_dir, sess, env, qnet, target_net, s_processor, p_copier, replay_memory_size=50000, burn_in_size=10000,
          target_update_iter=10, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_iter=500000,
          batch_size=32, hide_progress=False):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    loss_log = open(os.path.join(save_dir, "loss.csv"), 'w')
    loss_writer = csv.writer(loss_log, delimiter=',')
    rewards_log = open(os.path.join(save_dir, "rewards.csv"), 'w')
    rewards_writer = csv.writer(rewards_log, delimiter=',')

    loss_writer.writerow(['Iterations', 'Loss'])
    rewards_writer.writerow(['Iterations', 'Rewards'])

    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_iter)

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    replay_memory = []

    state = env.reset()
    state = s_processor.process(sess, state)
    state = np.stack([state] * 4, axis=2)
    for i in tqdm(range(burn_in_size), disable=hide_progress):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        next_state = s_processor.process(sess, next_state)
        next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
        replay_memory.append(Transition(state, action, reward, next_state, done))
        if done:
            state = env.reset()
            state = s_processor.process(sess, state)
            state = np.stack([state] * 4, axis=2)
        else:
            state = next_state

    train_iter = 0
    for train_ep in range(train_episodes):
        state = env.reset()
        state = s_processor.process(sess, state)
        state = np.stack([state] * 4, axis=2)

def main():
    env = gym.make("Breakout-v0")
    qnet = QNet(input_shape=[210, 160, 1], scope_name="QNet", lr=1e-3)
    target_net = QNet(input_shape=[210, 160, 1], scope_name="Target", lr=1e-3)
    sp = state_processor(input_shape=[210, 160, 3])
    pc = param_copier(qnet, target_net)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    start_time = str(datetime.now())
    print(start_time)
    train(10, "./" + start_time, sess, env, qnet, target_net, sp, pc)
    
if __name__ == '__main__':
    main()
