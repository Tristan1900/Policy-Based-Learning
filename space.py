#!/usr/bin/env python
import argparse
import pickle as p
import random
import sys
from time import strftime, localtime, time
import cv2
import gym
import numpy as np
from keras import optimizers, initializers
from keras.layers import Dense, Input
from keras.models import Model, load_model, Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation, Dropout, Flatten, Dense

class ReplayMemory:

    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory = []
        self.length = 0
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.full = False

    def sample_batch(self, batch_size=32):
        return random.sample(self.memory, batch_size)

    def append(self, transition):
        if self.full:
            self.memory[self.length] = transition
        else:
            self.memory.append(transition)
        self.length = (self.length + 1) % self.memory_size
        if self.length == 0:
            self.full = True


class QNetwork:

    def __init__(self, ns, na):
        self.model = Sequential()
        self.model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(84, 84, 4)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dense(na))
        adam = optimizers.Adam(lr=0.00001)
        self.model.compile(loss='mse', optimizer=adam)

        self.target_model = Sequential.from_config(self.model.get_config())

    def train(self, s, target):
        self.model.fit(s, target, batch_size=32, verbose=0)

    def train_target(self):
        w = self.model.get_weights()
        w_t = self.target_model.get_weights()
        length = len(w)
        for i in range(length):
            w_t[i] = 0.01 * w[i] + 0.99 * w_t[i]
        self.target_model.set_weights(w_t)

    def qvalues(self, s):
        return self.model.predict(s)

    def qvalues_target(self, s):
        return self.target_model.predict(s)

    def save_model(self, model_file):
        self.model.save(model_file)

    def load_model(self, model_file):
        self.model = load_model(model_file)
        self.target_model = load_model(model_file)

    def save_model_weights(self, weight_file):
        self.model.save_weights(weight_file)

    def load_model_weights(self, weight_file):
        self.model.set_weights(weight_file)
        self.target_model.set_weights(weight_file)


def load_agent(name):
    (env, gamma, ns, na, ss) = p.load(open(name + '.p', 'rb'))
    agent = DQNAgent(env, gamma)
    agent.net.load_model(name + '.h5')
    print("Agent loaded from {} created in {}".format(name + '.h5', ss))
    return agent


class DQNAgent:

    def __init__(self, env, gamma=1):
        self.env = env
        self.gamma = gamma
        self.ns = env.observation_space.shape[0]
        self.na = env.action_space.n
        self.net = QNetwork(self.ns, self.na)
        self.replay = ReplayMemory()
        self.buffer = []
        self.buffer_init()

    def buffer_init(self):
        self.env.reset()
        s, _, _, _ = self.env.step(0)
        self.buffer = [s,s,s,s]

    def save_agent(self, name):
        p.dump((self.env, self.gamma, self.ns, self.na, strftime('%Y-%m-%d %H:%M:%S', localtime(int(time())))),
               open(name + '.p', 'wb'))
        self.net.save_model(name + '.h5')

    def epsilon_greedy_policy(self, q_values, eps):
        k = random.uniform(0, 1)
        if k <= eps:
            a = random.randint(0, self.env.action_space.n - 1)
        else:
            a = np.argmax(q_values)
        return a

    def greedy_policy(self, q_values):
        return np.argmax(q_values)

    def convert_buffer(self):
        shape = (84,90)
        res = []
        t =[]
        for i in range(self.buffer):
            gray = cv2.cvtColor(self.buffer[i], cv2.COLOR_RGB2GRAY), shape)
            t[i] = cv2.resize(gray,shape)
        for i in range(t):
            res[i] = t[1:85, :, np.newaxis]
        return np.concatenate(res, axis=2)

    def train(self, case, steps, interval):
        self.burn_in_memory()
        print("after burning")
        iteration = 0
        episodes = 0
        while iteration < 100000:
            last = iteration
            self.env.reset()
            curr_state = self.convert_buffer()
            while iteration < 100000:
                eps = max(0.25 - (iteration / 100000) * 0.20, 0.05)
                s = self.convert_buffer()
                self.buffer = []

                q_values = self.net.qvalues(s.reshape(1, 84, 84, 4))
                action = self.epsilon_greedy_policy(q_values, eps)
                reward, done = 0, False

                for i in range(4):
                    t_observation, t_reward, t_done, _ = self.env.step(a)
                    reward += t_reward
                    if t_done == True
                        done = True
                    self.buffer.append(t_observation)

                s_ = self.convert_buffer()
                if done:
                    s_ = None

                trans = (s, action, reward, s_)
                self.replay.append(trans)

                batch = self.replay.sample_batch()

                p = self.net.qvalues(np.array([i[0].reshape(84, 84, 4) for i in batch]))
                p_ = self.net.qvalues(np.array([(i[3].reshape(84, 84, 4) if i[3] is not None else np.zeros(shape=(84,84,4))) for i in batch]))

                p_target = self.net.qvalues_target(np.array([i[0].reshape(84, 84, 4) for i in batch]))
                
                p_target_ = self.net.qvalues_target(np.array([(i[3].reshape(84, 84, 4) if i[3] is not None else np.zeros(shape=(84,84,4))) for i in batch]))
                x = np.zeros((len(batch), 84,84,4))
                y = np.zeros((len(batch), self.na))

                for i, val in enumerate(batch):
                    s1 = val[0]
                    a1 = val[1]
                    r1 = val[2]
                    s_1 = val[3]

                    if s_1 is None:
                        p[i][a1] = r1
                    else:
                        p[i][a1] = r1 + self.gamma * np.max(p_target_[i])

                    x[i] = s1
                    y[i] = p[i]
                self.net.train(x, y)
                self.net.train_target()
                # s = s_

                iteration += 1
                if iteration % 1000 == 0:
                    print(iteration)
                if done:
                    break
            episodes += 1
            if episodes % interval == 0:
                print("The {}th episodes".format(episodes))
                name = "weight" + str(episodes)
                self.net.save_model(name)
                self.test()

    def test(self, episodes=20, render=True):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.
        rewards = 0
        for _ in range(episodes):
            s = self.env.reset()
            steps = 0
            s1, r1, _, _ = self.env.step(0)
            s2, r2, _, _ = self.env.step(0)
            s3, r3, _, _ = self.env.step(0)
            s4, r4, _, _ = self.env.step(0)
            self.buffer = [s1,s2,s3,s4]
            s = self.convert_buffer()
            while True:
                if render:
                    self.env.render()
                s = self.convert_buffer()
                self.buffer = []
                a = self.greedy_policy(self.net.qvalues(s.reshape(1, 84, 84, 4)))
                reward, done = 0, False
                for i in range(4):
                    t_observation, t_reward, t_done, _ = self.env.step(a)
                    reward += t_reward
                    if t_done == True
                        done = True
                    self.buffer.append(t_observation)
                rewards += reward
                if done:
                    break
        print("The average reward of {} episodes is {}".format(episodes, rewards / episodes))
        f = open("stat.dat","a")
        f.write("{}\n".format(rewards / episodes))
        f.close()
        return rewards / episodes

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        print("buring in")
        iteration = 0
        while iteration <= self.replay.burn_in:
            s = self.env.reset()
            while True:
                s = self.convert_buffer()
                self.buffer = []

                reward, done = 0, False
                a = self.epsilon_greedy_policy(self.net.qvalues(s.reshape(1, 84, 84, 4)),1)
                for i in range(4):
                    t_observation, t_reward, t_done, _ = self.env.step(a)
                    reward += t_reward
                    if t_done == True
                        done = True
                    self.buffer.append(t_observation)

                s_ = self.convert_buffer()    
                if done:
                    s_ = None
                transitions = (s, a, reward, s_)
                self.replay.append(transitions)
                # s = s_
                iteration += 1
                if iteration % 100 == 0:
                    print(iteration)
                if done:
                    break


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=str)
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--model', dest='model_file', type=str)
    return parser.parse_args()

def main(args):
    env = gym.make("SpaceInvaders-v0")
    agent = DQNAgent(env, gamma=0.99)
    agent.train(case='eps', steps=10, interval=33333)

if __name__ == '__main__':
    main(sys.argv)
