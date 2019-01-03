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
import matplotlib.pyplot as plt

from collections import namedtuple

class state_processor():
    def __init__(self, input_shape, output_shape=None, scope_name="state_processor"):
        self.scope_name = scope_name
        with tf.variable_scope(self.scope_name):
            self.input_state = tf.placeholder(shape=input_shape, dtype=tf.float32)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            if output_shape is not None:
                self.output = tf.image.resize_images(self.output, output_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.dtypes.cast(self.output, tf.uint8)

    def process(self, sess, input_state):
        processed_state = sess.run(self.output, feed_dict={self.input_state: input_state})
        return processed_state

class QNet():
    def __init__(self, scope_name, input_shape, lr=1e-3, momentum=0.95, num_actions=4, trainable=True):
        self.scope_name = scope_name
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.trainable = trainable
        self.momentum = momentum
        if self.trainable:
            self.lr = lr
        self.history_size = input_shape[-1]
        with tf.variable_scope(scope_name):
            self.build_model()
        if self.trainable:
            with tf.variable_scope("optim" + scope_name):

                self.loss = tf.losses.huber_loss(self.y, self.action_preds)

                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, momentum=self.momentum)
                self.model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_name)
                self.update_op = self.optimizer.minimize(self.loss, var_list=self.model_vars)

    def build_model(self):
        self.X = tf.placeholder(shape=[None] + list(self.input_shape), dtype=tf.float32, name="X")
        self.y = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        self.X_rescaled = tf.divide(self.X, 255.0)

        bs = tf.shape(self.X)[0]
        self.conv1 = tf.contrib.layers.conv2d(self.X_rescaled, 32, 8, 4, activation_fn=tf.nn.relu,
                                              weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        self.conv2 = tf.contrib.layers.conv2d(self.conv1, 64, 4, 2, activation_fn=tf.nn.relu,
                                              weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        self.conv3 = tf.contrib.layers.conv2d(self.conv2, 64, 3, 1, activation_fn=tf.nn.relu,
                                              weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        #  self.pool3 = tf.contrib.layers.max_pool2d(self.conv3, 2, padding='SAME')

        self.flattened = tf.contrib.layers.flatten(self.conv3)
        self.fc1 = tf.contrib.layers.fully_connected(self.flattened, 512, activation_fn=tf.nn.relu,
                                              weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        self.preds = tf.contrib.layers.fully_connected(self.fc1, self.num_actions, activation_fn=None,
                                              weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        self.indices = tf.stack([tf.range(bs), self.actions], axis=1)
        self.action_preds = tf.gather_nd(self.preds, self.indices)

    def predict(self, sess, state):
        return sess.run(self.preds, feed_dict={self.X: state})

    def update(self, sess, state, actions, y):
        loss, _ = sess.run([self.loss, self.update_op], feed_dict={self.X: state,
                                                                   self.y: y,
                                                                   self.actions: actions})
        return loss

class param_copier():
    def __init__(self, qnet, target_net):
        qnet_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=qnet.scope_name)
        self.qnet_params = sorted(qnet_params, key=lambda v: v.name)
        target_net_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_net.scope_name)
        self.target_net_params = sorted(target_net_params, key=lambda v: v.name)

        self.copy_ops = []
        for q_v, t_v in zip(qnet_params, target_net_params):
            cp = t_v.assign(q_v)
            self.copy_ops.append(cp)

        self.check_ops = []
        for q_v, t_v in zip(qnet_params, target_net_params):
            ch = tf.math.reduce_max(tf.math.abs(q_v - t_v))
            self.check_ops.append(ch)

    def copy(self, sess):
        sess.run(self.copy_ops)

    def check(self, sess, epsilon=1e-6):
        diff = sess.run(self.check_ops)
        return max(diff) < epsilon

def epsilon_greedy(qnet, sess, state, epsilon, num_actions=4):
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)
    else:
        q_values = qnet.predict(sess, np.expand_dims(state, 0))[0]
        return np.argmax(q_values)


def reset_env(env, s_processor, sess):
    state = env.reset()
    state = s_processor.process(sess, state)
    state = np.stack([np.squeeze(state, axis=2)] * 4, axis=2)
    return state

def check_preprocessing(env, s_processor, sess):
    state = env.reset()
    state = s_processor.process(sess, state)
    plt.imshow(state[:, :, 0], cmap='gray')
    plt.show()
    for action in range(1, env.action_space.n):
        next_state, reward, done, _ = env.step(action)
        next_state = s_processor.process(sess, next_state)
        plt.imshow(next_state[:, :, 0], cmap='gray')
        plt.show()

def evaluate(eval_episodes, sess, env, qnet, s_processor, epsilon=0.05):
    rewards = []
    for ep in range(eval_episodes):
        state = reset_env(env, s_processor, sess)
        episode_reward = 0
        done = False
        while not done:
            action =  epsilon_greedy(qnet, sess, state, epsilon)
            next_state, reward, done, _ = env.step(action)
            next_state = s_processor.process(sess, next_state)
            next_state = np.append(state[:, :, 1:], next_state, axis=2)
            episode_reward += reward
        rewards.append(episode_reward)
    return rewards

def train(train_episodes, save_dir, sess, env, qnet, target_net, s_processor, p_copier, replay_memory_size=50000, burn_in=10000,
          target_update_iter=10, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_iter=500000,
          eval_every=20, eval_episodes=10, batch_size=32, hide_progress=False, use_double=False):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "params"), 'w') as param_file:
        layers = [x for x in dir(qnet) if x.startswith("conv") or x.startswith("fc") or x.startswith("preds")]
        for v in layers:
            attr = getattr(qnet, v)
            param_file.write(str(attr) + "\n")
        param_file.write("input shape " + str(qnet.input_shape[0]) + " " + str(qnet.input_shape[1]) + "\n")
        param_file.write("Learning rate " + str(qnet.lr) + "\n")
        param_file.write("memory size " + str(replay_memory_size) + "\n")
        param_file.write("burn in " + str(burn_in) + "\n")
        param_file.write("target update " + str(target_update_iter) + "\n")
        param_file.write("gamma " + str(gamma) + "\n")
        param_file.write("epsilon start " + str(epsilon_start) + "\n")
        param_file.write("epsilon end " + str(epsilon_end) + "\n")
        param_file.write("epsilon decay " + str(epsilon_decay_iter) + "\n")
        param_file.write("eval every " + str(eval_every) + "\n")
        param_file.write("eval episodes " + str(eval_episodes) + "\n")
        param_file.write("Double? " + str(use_double) + "\n")
    loss_log = open(os.path.join(save_dir, "loss.csv"), 'w')
    loss_writer = csv.writer(loss_log, delimiter=',')
    rewards_log = open(os.path.join(save_dir, "train_rewards.csv"), 'w')
    rewards_writer = csv.writer(rewards_log, delimiter=',')
    eval_log = open(os.path.join(save_dir, "eval_rewards.csv"), 'w')
    eval_writer = csv.writer(eval_log, delimiter=',')

    loss_writer.writerow(['Iterations', 'Loss'])
    rewards_writer.writerow(['Episode', 'Reward'])
    eval_writer.writerow(['Episode', 'Average Reward', 'Std. Reward', 'Min Reward', 'Max Reward'])

    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_iter)

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    replay_memory = []

    state = reset_env(env, s_processor, sess)
    #  print(state.shape)
    #  a = sess.run([qnet.pool1, qnet.pool2, qnet.pool3], {qnet.X : np.expand_dims(state, 0)})
    #  for x in a:
        #  print(x.shape)
    #  return
    for i in tqdm(range(burn_in), disable=hide_progress):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        next_state = s_processor.process(sess, next_state)
        next_state = np.append(state[:, :, 1:], next_state, axis=2)
        replay_memory.append(Transition(state, action, reward, next_state, done))
        if done:
            state = reset_env(env, s_processor, sess)
        else:
            state = next_state

    train_iter = 0
    for train_ep in tqdm(range(train_episodes), disable=hide_progress):
        if train_ep % eval_every == 0:
            eval_rewards = evaluate(eval_episodes, sess, env, qnet, s_processor, epsilon=epsilon_end)
            eval_mean = np.mean(eval_rewards)
            eval_std = np.std(eval_rewards)
            eval_writer.writerow([train_ep, eval_mean, eval_std, min(eval_rewards), max(eval_rewards)])
            eval_log.flush()
        state = reset_env(env, s_processor, sess)
        episode_reward = 0
        done = False
        while not done:
            epsilon = epsilons[min(train_iter, epsilon_decay_iter-1)]
            if train_iter % target_update_iter == 0:
                p_copier.copy(sess)
            action = epsilon_greedy(qnet, sess, state, epsilon)
            next_state, reward, done, _ = env.step(action)
            next_state = s_processor.process(sess, next_state)
            next_state = np.append(state[:, :, 1:], next_state, axis=2)

            episode_reward += reward

            replay_memory.append(Transition(state, action, reward, next_state, done))
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            train_batch = random.sample(replay_memory, batch_size)
            train_states, train_actions, train_rewards, train_next_states, train_done = map(np.array, zip(*train_batch))

            if use_double:
                q_values_next = qnet.predict(sess, train_next_states)
                actions = np.argmax(q_values_next, axis=1)
                target_values_next = target_net.predict(sess, train_next_states)
                target_values_next = target_values_next[np.arange(batch_size), actions]
            else:
                target_values_next = target_net.predict(sess, train_next_states)
                target_values_next = np.max(target_values_next, axis=1)

            train_targets = train_rewards + (1 - train_done.astype(np.float32)) * gamma * target_values_next

            loss = qnet.update(sess, train_states, train_actions, train_targets)

            loss_writer.writerow([train_iter, loss])
            train_iter += 1

            state = next_state
        rewards_writer.writerow([train_ep, episode_reward])
        rewards_log.flush()
    eval_rewards = evaluate(eval_episodes, sess, env, qnet, s_processor, epsilon=epsilon_end)
    eval_mean = np.mean(eval_rewards)
    eval_std = np.std(eval_rewards)
    eval_writer.writerow([train_ep, eval_mean, eval_std, min(eval_rewards), max(eval_rewards)])
    eval_log.flush()

def main():
    env = gym.make("Breakout-v0")

    history_size = 4
    observation_shape = list(env.observation_space.shape)
    state_shape = [84, 84]
    sp = state_processor(input_shape=observation_shape, output_shape=state_shape)

    qnet = QNet(input_shape=state_shape + [history_size], scope_name="QNet", lr=2.5e-4)
    target_net = QNet(input_shape=state_shape + [history_size], scope_name="Target", trainable=False)

    pc = param_copier(qnet, target_net)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    start_time = str(datetime.now())
    print(start_time)

    train(1, "./logs/" + start_time, sess, env, qnet, target_net, sp, pc, hide_progress=False, target_update_iter=10000, burn_in=1000, replay_memory_size=50000, eval_every=50, eval_episodes=2, use_double=True, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_iter=500000)
    
if __name__ == '__main__':
    main()
