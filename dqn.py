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
    def __init__(self, scope_name, input_shape, lr=1e-4, decay=0.9, momentum=0, num_actions=4, trainable=True):
        self.scope_name = scope_name
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.trainable = trainable
        self.momentum = momentum
        self.lr = lr
        self.decay = decay
        self.history_size = input_shape[-1]
        with tf.variable_scope(scope_name):
            self.build_model()
        if self.trainable:
            with tf.variable_scope("optim" + scope_name):
                self.loss = tf.losses.huber_loss(self.y, self.action_preds)

                self.optimizer = tf.train.RMSPropOptimizer(self.lr, self.decay, self.momentum)
                self.model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_name)
                self.saver = tf.train.Saver(self.model_vars)
                self.update_op = self.optimizer.minimize(self.loss, var_list=self.model_vars)
                self.reset_optimizer = tf.variables_initializer([self.optimizer.get_slot(var, name) for name in self.optimizer.get_slot_names() for var in self.model_vars])

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

    def save_model(self, sess, path="./models", global_step=0):
        self.saver.save(sess, path, global_step)

    def load_model(self, sess, path):
        self.saver.restore(sess, path)

class param_copier():
    def __init__(self, qnet, target_net):
        self.qnet = qnet
        self.target_net = target_net
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


def reset_env(env, s_processor, sess, history_size=4):
    state = env.reset()
    state = s_processor.process(sess, state)
    state = np.stack([np.squeeze(state, axis=2)] * history_size, axis=2)
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

def evaluate(eval_episodes, sess, env_name, qnet, s_processor, history_size, epsilon=0.05):
    env = gym.make(env_name)
    rewards = []
    for ep in range(eval_episodes):
        state = reset_env(env, s_processor, sess, history_size)
        lives = env.env.ale.lives()
        episode_reward = 0
        done = False
        while not done:
            action =  epsilon_greedy(qnet, sess, state, epsilon)
            next_state, reward, done, info = env.step(action)
            next_state = s_processor.process(sess, next_state)
            if info['ale.lives'] < lives:
                lives = info['ale.lives']
                state = np.stack([next_state[:, :, -1]] * history_size, axis=2)
            else:
                state = np.append(state[:, :, 1:], next_state, axis=2)
            episode_reward += reward
        rewards.append(episode_reward)
    return rewards

def train(train_iters, save_dir, sess, env, qnet, target_net, s_processor, p_copier, replay_memory_size=50000, burn_in=10000,
          target_update_iter=10, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_iter=500000,
          eval_every=100, eval_episodes=100, eval_epsilon=0.05, model_prefix="dqn", history_size=4, batch_size=32,
          hide_progress=False, use_double=False):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_dir = os.path.join(save_dir, "models/")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    with open(os.path.join(save_dir, "params"), 'w') as param_file:
        layers = [x for x in dir(qnet) if x.startswith("conv") or x.startswith("fc") or x.startswith("preds")]
        for v in layers:
            attr = getattr(qnet, v)
            param_file.write(str(attr) + "\n")
        param_file.write("input shape " + str(qnet.input_shape[0]) + " " + str(qnet.input_shape[1]) + "\n")
        param_file.write("memory size " + str(replay_memory_size) + "\n")
        param_file.write("burn in " + str(burn_in) + "\n")
        param_file.write("target update " + str(target_update_iter) + "\n")
        param_file.write("gamma " + str(gamma) + "\n")
        param_file.write("epsilon start " + str(epsilon_start) + "\n")
        param_file.write("epsilon end " + str(epsilon_end) + "\n")
        param_file.write("epsilon decay " + str(epsilon_decay_iter) + "\n")
        param_file.write("eval every " + str(eval_every) + "\n")
        param_file.write("eval episodes " + str(eval_episodes) + "\n")
        param_file.write("eval epsilon " + str(eval_epsilon) + "\n")
        param_file.write("ddqn? " + str(use_double) + "\n")
    loss_log = open(os.path.join(save_dir, "loss.csv"), 'w')
    loss_writer = csv.writer(loss_log, delimiter=',')
    train_log = open(os.path.join(save_dir, "train_log.csv"), 'w')
    train_writer = csv.writer(train_log, delimiter=',')
    eval_log = open(os.path.join(save_dir, "eval_rewards.csv"), 'w')
    eval_writer = csv.writer(eval_log, delimiter=',')

    loss_writer.writerow(['Iterations', 'Loss'])
    train_writer.writerow(['Episode', 'Reward', 'Length'])
    eval_writer.writerow(['Episode', 'Average Reward', 'Std. Reward', 'Min Reward', 'Max Reward'])

    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_iter)

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    replay_memory = []

    state = reset_env(env, s_processor, sess, history_size)
    lives = env.env.ale.lives()
    for i in tqdm(range(burn_in), disable=hide_progress):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        next_state = s_processor.process(sess, next_state)
        next_state = np.append(state[:, :, 1:], next_state, axis=2)
        replay_memory.append(Transition(state, action, np.clip(reward, -1, 1), next_state, info['ale.lives'] < lives))
        if info['ale.lives'] < lives:
            lives = info['ale.lives']
            next_state = np.stack([next_state[:, :, -1]] * history_size, axis=2)

        if done:
            state = reset_env(env, s_processor, sess, history_size)
            lives = env.env.ale.lives()
        else:
            state = next_state

    state = reset_env(env, s_processor, sess, history_size)
    lives = env.env.ale.lives()
    episode_reward = 0
    episode_length = 0
    train_episode = 0
    for train_iter in tqdm(range(train_iters), disable=hide_progress):
        if train_iter % eval_every == 0:
            eval_rewards = evaluate(eval_episodes, sess, env.unwrapped.spec.id, qnet, s_processor, history_size, epsilon=eval_epsilon)
            eval_mean = np.mean(eval_rewards)
            eval_std = np.std(eval_rewards)
            eval_writer.writerow([train_episode, eval_mean, eval_std, min(eval_rewards), max(eval_rewards)])
            eval_log.flush()
            qnet.save_model(sess, model_dir + model_prefix, train_iter)

        if train_iter % target_update_iter == 0:
            p_copier.copy(sess)

        epsilon = epsilons[min(train_iter, epsilon_decay_iter-1)]
        action = epsilon_greedy(qnet, sess, state, epsilon)
        next_state, reward, done, info = env.step(action)
        next_state = s_processor.process(sess, next_state)
        next_state = np.append(state[:, :, 1:], next_state, axis=2)

        episode_reward += reward
        episode_length += 1

        replay_memory.append(Transition(state, action, np.clip(reward, -1, 1), next_state, info['ale.lives'] < lives))
        if info['ale.lives'] < lives:
            lives = info['ale.lives']
            next_state = np.stack([next_state[:, :, -1]] * history_size, axis=2)

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

        if done:
            train_writer.writerow([train_episode, episode_reward, episode_length])
            train_log.flush()
            state = reset_env(env, s_processor, sess, history_size)
            episode_reward = 0
            episode_length = 0
            train_episode += 1
        else:
            state = next_state

def main():
    env_name = "BreakoutDeterministic-v4"
    env = gym.make(env_name)
    history_size = 4
    observation_shape = list(env.observation_space.shape)
    state_shape = [84, 84]
    sp = state_processor(input_shape=observation_shape, output_shape=state_shape)

    qnet = QNet(input_shape=state_shape + [history_size], scope_name="QNet", lr=2.5e-4)
    target_net = QNet(input_shape=state_shape + [history_size], scope_name="Target", trainable=False)

    pc = param_copier(qnet, target_net)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    model_prefix = "dqn"
    start_time = str(datetime.now())
    print(start_time)

    train(int(5e7), "./logs/" + model_prefix + "-" + start_time, sess, env, qnet, target_net, sp, pc, model_prefix=model_prefix, eval_episodes=100, hide_progress=False, target_update_iter=10000, burn_in=50000, replay_memory_size=int(1e6), eval_every=int(1e6), eval_epsilon=0.01, use_double=False, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_iter=int(1e6), history_size=history_size)
    
if __name__ == '__main__':
    main()
