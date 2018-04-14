import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reinforce import Reinforce

class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self, env, lr, critic_lr, render, n=20):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - n: The value of N in N-step A2C.
        self.log = True
        self.render = render
        self.env = env
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.lr = lr
        self.critic_lr = critic_lr
        self.n = n
        self.define_actor()
        self.define_critic()
        self.num_episodes = 30000
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def define_actor(self):
        self.actor_state = tf.placeholder(tf.float32, [None, self.state_space])
        self.actor_action = tf.placeholder(tf.int32, [None, ])
        self.actor_reward = tf.placeholder(tf.float32, [None, ])

        layer1 = tf.layers.dense(
            inputs=self.actor_state,
            units=16,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.glorot_uniform(),
            bias_initializer=tf.constant_initializer(0),
        )

        layer2 = tf.layers.dense(
            inputs=layer1,
            units=16,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.glorot_uniform(),
            bias_initializer=tf.constant_initializer(0),
        )

        layer3 = tf.layers.dense(
            inputs=layer2,
            units=16,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.glorot_uniform(),
            bias_initializer=tf.constant_initializer(0),
        )

        layer4 = tf.layers.dense(
            inputs=layer3,
            units=self.action_space,
            activation=None,
            kernel_initializer=tf.keras.initializers.glorot_uniform(),
            bias_initializer=tf.constant_initializer(0),
        )
        self.actor_predict = tf.nn.softmax(layer4)
        my_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=layer4, labels=self.actor_action)
        loss = tf.reduce_mean(my_loss * self.actor_reward)
        self.actor_train = tf.train.AdamOptimizer(self.lr).minimize(loss)  

    def define_critic(self):
        self.critic_state = tf.placeholder(tf.float32, [None, self.state_space])
        self.critic_r = tf.placeholder(tf.float32, [None, ])

        layer1 = tf.layers.dense(
            inputs=self.critic_state,
            units=16,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.glorot_uniform(),
            bias_initializer=tf.constant_initializer(0),
        )

        layer2 = tf.layers.dense(
            inputs=layer1,
            units=16,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.glorot_uniform(),
            bias_initializer=tf.constant_initializer(0),
        )

        layer3 = tf.layers.dense(
            inputs=layer2,
            units=16,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.glorot_uniform(),
            bias_initializer=tf.constant_initializer(0),
        )

        layer4 = tf.layers.dense(
            inputs=layer3,
            units=1,
            activation=None,
            kernel_initializer=tf.keras.initializers.glorot_uniform(),
            bias_initializer=tf.constant_initializer(0),
        )
        self.critic_predict = layer4
        # loss = tf.reduce_mean(tf.square(self.critic_r - layer4))
        loss = tf.losses.mean_squared_error(labels=self.critic_r, predictions=tf.reshape(layer4, shape=[-1]))
        self.critic_train= tf.train.AdamOptimizer(self.critic_lr).minimize(loss)

    def train(self, gamma=0.99):
        # Trains the model on a single episode using A2C.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        for counter in range(self.num_episodes):
            s, a, r = self.generate_episode()
            r = [item/200 for item in r] 
            dr = np.zeros_like(r, dtype=np.float32)
            for i in range(len(r) - 1, -1, -1):
                for j in range(self.n):
                    if i + j < len(r):
                        dr[i] += r[i + j]*(gamma**j)
                    else:
                        break
            for x in range(5):
                v = self.sess.run(self.critic_predict, feed_dict={self.critic_state: np.reshape(s, (-1, self.state_space))})
                v = v.flatten()
                new_r = np.zeros_like(dr, dtype=np.float32)
                for y in range(len(dr)):
                    if self.n + y < len(r):
                        new_r[y] = dr[y] + gamma**self.n*v[self.n + y]
                    else:
                        new_r[y] = dr[y]
                reward = new_r - v
                self.sess.run(self.actor_train, feed_dict={
                    self.actor_state: np.reshape(np.array(s, dtype=np.float32),(-1, self.state_space)),
                    self.actor_action: np.array(a, dtype=np.float32),
                    self.actor_reward: np.array(reward, dtype=np.float32),
                })
                self.sess.run(self.critic_train, feed_dict={
                    self.critic_state: np.reshape(np.array(s, dtype=np.float32),(-1, self.state_space)),
                    self.critic_r: new_r,
                })
  
            if counter % 500 == 0:
                self.test()
                self.save()
        return 

    def generate_episode(self):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        states = []
        actions = []
        rewards = []
        s = self.env.reset()
        while True:
            states.append(s)
            a = self.sess.run(self.actor_predict, feed_dict={self.actor_state: np.reshape(s, (-1, self.state_space))})
            a =  np.random.choice(range(a.shape[1]), p=a[0])
            actions.append(a)
            s, r, done, _ = self.env.step(a)
            rewards.append(r)
            if done:
                break
        return states, actions, rewards

    def test(self, episode=100):
        res = []
        for i in range(episode):
            reward_tmp = 0
            state = self.env.reset()
            while True:
                action = self.sess.run(self.actor_predict, feed_dict={self.actor_state: np.reshape(state, (-1, self.state_space))})
                if self.render:
                    self.env.render()
                action =  np.random.choice(range(action.shape[1]), p=action[0])
                state, reward, done, _ = self.env.step(action)
                reward_tmp += reward
                if done:
                    break
            res.append(reward_tmp)
        print("Test mean reward is {}".format(np.mean(res)))
        print("Test std reward is {}".format(np.std(res)))
        if self.log:
            f = open("./stat","a+") 
            f.write("mean/std {} ".format(np.mean(res)))
            f.write("{}\n".format(np.std(res)))
        return


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the actor model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=1e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=100, help="The value of N in N-step A2C.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    critic_lr = args.critic_lr
    n = args.n
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
    # env = gym.make('CartPole-v0')
    # Load the actor model from file.

    # TODO: Train the model using A2C and plot the learning curves.
    a2c = A2C(env=env, lr=lr, critic_lr=critic_lr, render=render)
    a2c.train()


if __name__ == '__main__':
    main(sys.argv)
