import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym
from keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, env, lr, num_episodes, render):
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.lr = lr
        self.num_episodes = num_episodes
        self.render = render
        self.env = env
        self.define_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def define_net(self):
        self.tf_state = tf.placeholder(tf.float32, [None, self.state_space])
        self.tf_action = tf.placeholder(tf.int32, [None, ])
        self.tf_reward = tf.placeholder(tf.float32, [None, ])

        layer1 = tf.layers.dense(
            inputs=self.tf_state,
            units=16,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.glorot_uniform(),
            bias_initializer=tf.constant_initializer(0),
            name='l1'
        )

        layer2 = tf.layers.dense(
            inputs=layer1,
            units=16,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.glorot_uniform(),
            bias_initializer=tf.constant_initializer(0),
            name='l2'
        )

        layer3 = tf.layers.dense(
            inputs=layer2,
            units=16,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.glorot_uniform(),
            bias_initializer=tf.constant_initializer(0),
            name='l3'
        )

        layer4 = tf.layers.dense(
            inputs=layer3,
            units=self.action_space,
            activation=None,
            kernel_initializer=tf.keras.initializers.glorot_uniform(),
            bias_initializer=tf.constant_initializer(0),
            name='l4'
        )
        self.predict = tf.nn.softmax(layer4)
        my_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=layer4, labels=self.tf_action)
        loss = tf.reduce_mean(my_loss * self.tf_reward)
        self.train_model = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def save(self):
        self.saver = tf.train.Saver()
        self.save_path = self.saver.save(self.sess, "./save/model.ckpt")
        print("Model saved in path: %s" % self.save_path)

    def restore(self):
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, "./save/model.ckpt")
        print("Model restored.")

    def train(self, gamma=0.99):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        epi = 0
        for i in range(self.num_episodes):
            s = []
            a = []
            r = []
            for i in range(5):
                s_t, a_t, r_t = self.generate_episode()
                dr = []
                dr = np.zeros_like(r_t)
                for i in range(len(r_t) - 1, -1, -1):
                    if i == len(r_t) - 1:
                        dr[i] = r_t[i]
                        continue
                    dr[i] = r_t[i] + gamma * dr[i + 1]
                dr /= 200
                s.extend(s_t)
                a.extend(a_t)
                r.extend(dr)
            self.sess.run(self.train_model, feed_dict={
                self.tf_state: np.reshape(s,(-1, self.state_space)),
                self.tf_action: np.array(a),
                self.tf_reward: r,
            })
            epi = epi + 1
            if epi % 100 == 0:
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
            a = self.sess.run(self.predict, feed_dict={self.tf_state: np.reshape(s, (-1, self.state_space))})
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
                action = self.sess.run(self.predict, feed_dict={self.tf_state: np.reshape(state, (-1, self.state_space))})
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
        f = open("./stat","a+") 
        f.write("mean/std {} ".format(np.mean(res)))
        f.write("{}\n".format(np.std(res)))
        return


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=30000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=0.0005, help="The learning rate.")

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
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
    # env = gym.make('CartPole-v0')
    # TODO: Train the model using REINFORCE and plot the learning curve.
    rein = Reinforce(env, lr=lr, num_episodes=num_episodes, render=render)
    # rein.restore()
    rein.train()
#    rein.restore()
#    rein.test()


if __name__ == '__main__':
    main(sys.argv)
