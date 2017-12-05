"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf
import math

# reproducible
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
            batch_size=16,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.epsilon = 1e-7

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.prob = [0.0] * self.n_actions

        print ('init of brain4')

        self._build_net()

        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=100)

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=200,
            activation=tf.nn.selu,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        self.all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        self.all_act_prob = tf.nn.softmax(self.all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob + self.epsilon)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_sum(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    def choose_action(self, x):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: x[np.newaxis, :]})
        for weight in prob_weights:
            for i in range(self.n_actions):
                self.prob[i] += weight[i]

        if math.isnan(self.prob[i]):
            print ('Nan: ', prob_weights)

        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def getProbs(self):
        prob = self.prob
        self.prob = [0.0] * self.n_actions

        return prob

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def checkActDist(self):
        actions = np.zeros([self.n_actions])
        for a in self.ep_as:
            actions[a] += 1

        return actions

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            if self.ep_rs[t] != 0.0:
                running_add = 0

            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std( discounted_ep_rs)
        return discounted_ep_rs

    def save(self, path, gStep):
        self.saver.save(self.sess, path, global_step=gStep)

    def restore(self, path):
        self.saver.restore(self.sess, path)

