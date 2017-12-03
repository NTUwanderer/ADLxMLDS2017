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
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.epsilon = 1e-7
        self.entropy = 0.01
        self.norm = 50

        self.ep_obs, self.ep_as, self.ep_rs, self.ep_vs = [], [], [], []

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
        self.states = tf.placeholder(tf.float32, shape=[None, *self.n_features], name="states")
        self.actions = tf.placeholder(tf.uint8, shape=[None], name="actions")
        action_onehots = tf.one_hot(self.actions, depth=self.n_actions, name="action_onehots")
        self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
        self.advantages = tf.placeholder(tf.float32, shape=[None], name="advantages")

        net = self.states

        with tf.variable_scope("layer1"):
            net = tf.layers.conv2d(net, filters=16, kernel_size=(8, 8), strides=(4, 4), name="conv")
            net = tf.nn.relu(net, name="relu")

        with tf.variable_scope("layer2"):
            net = tf.layers.conv2d(net, filters=32, kernel_size=(4, 4), strides=(2, 2), name="conv")
            net = tf.nn.relu(net, name="relu")

        net = tf.contrib.layers.flatten(net)

        with tf.variable_scope("fc1"):
            net = tf.layers.dense(net, units=256, name="fc")
            net = tf.nn.relu(net, name="relu")

        with tf.variable_scope("action_network"):
            action_scores = tf.layers.dense(net, units=self.n_actions, name="action_scores")
            self.action_probs = tf.nn.softmax(action_scores, name="action_probs")
            single_action_prob = tf.reduce_sum(self.action_probs * action_onehots, axis=1)
            log_action_prob = - tf.log(single_action_prob + self.epsilon) * self.advantages
            action_loss = tf.reduce_sum(log_action_prob)

        with tf.variable_scope("entropy"):
            entropy = - tf.reduce_sum(self.action_probs * tf.log(self.action_probs + self.epsilon), axis=1)
            entropy_sum = tf.reduce_sum(entropy)

        with tf.variable_scope("value_network"):
            self.values = tf.squeeze(tf.layers.dense(net, units=1, name="values"))
            value_loss = tf.reduce_sum(tf.squared_difference(self.rewards, self.values))

        with tf.variable_scope("total_loss"):
            self.loss = action_loss + value_loss * 0.5 - entropy_sum * self.entropy

        with tf.variable_scope("train_op"):
            self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            gradients = self.optim.compute_gradients(loss=self.loss)
            gradients = [(tf.clip_by_norm(grad, self.norm), var) for grad, var in gradients]
            self.train_op = self.optim.apply_gradients(gradients,
                                                       global_step=tf.train.get_or_create_global_step())

    def get_actions(self, states):
        """Get actions given states
        Args:
            states (4-D Array): States Array of shape (N, H, W, C)
        
        Returns:
            actions (1-D Array): Action Array of shape (N,)
        """
        feed = {
            self.states: np.reshape(states, [-1, *self.input_shape])
        }
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: states[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def get_values(self, stats):
        """Get values given states
        Args:
            states (4-D Array): States Array of shape (N, H, W, C)
        
        Returns:
            values (1-D Array): Values (N,)
        """
        feed = {
            self.states: np.reshape(states, [-1, *self.input_shape])
        }
        return self.sess.run(self.values, feed).reshape(-1)

    def get_actions_values(self, states):
        """Get actions and values given states
        
        Args:
            states (4-D Array): States Array of shape (N, H, W, C)
        
        Returns:
            actions (1-D Array): Action Array of shape (N,)
            values (1-D Array): Values (N,)
        """
        feed = {
            self.states: states,
        }

        action_probs, values = self.sess.run([self.action_probs, self.values], feed)
        noises = np.random.uniform(size=action_probs.shape[0])[:, np.newaxis]

        return (np.cumsum(action_probs, axis=1) > noises).argmax(axis=1), values.flatten()

    def add_episode(self):
        self.ep_obs.append([])
        self.ep_as.append([])
        self.ep_rs.append([])
        self.ep_vs.append([])

    def store_transition(self, s, a, r, v):
        self.ep_obs[-1].append(s)
        self.ep_as[-1].append(a)
        self.ep_rs[-1].append(r)
        self.ep_vs[-1].append(v)

    def checkActDist(self):
        actions = np.zeros([self.n_actions])
        for a in self.ep_as:
            actions[a] += 1

        return actions

    def learn(self):
        states = np.vstack([s for s in self.ep_obs if len(s) > 0])
        actions = np.hstack(self.ep_as)
        values = np.hstack(self.ep_vs)
        # discount and normalize episode reward
        rewards = self._discount_multi_rewards()
        rewards = np.hstack(rewards)
        rewards -= np.mean(rewards)
        rewards /= np.std(rewards) + self.epsilon

        advantages = rewards - values
        advantages -= np.mean(advantages)
        advantages /= np.std(advantages) + self.epsilon

        feed = {
            self.states: states,
            self.actions: actions,
            self.rewards: rewards,
            self.advantages: advantages
        }

        _, global_step = self.sess.run([self.train_op,
                                               tf.train.get_global_step()],
                                              feed_dict=feed)
        # self.summary_writer.add_summary(summary_op, global_step=global_step)
        
        self.ep_obs, self.ep_as, self.ep_rs, self.ep_vs = [], [], [], []


    def _discount_rewards(self, rewards):
        discounted = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0

        for i in reversed(range(len(rewards))):
            if rewards[i] != 0:
                running_add = 0
            running_add = rewards[i] + self.gamma * running_add
            discounted[i] = running_add
        return discounted


    def _discount_multi_rewards(self):
        # discount episode rewards
        discounted_ep_rs = []
        for r in self.ep_rs:
            discounted_ep_rs.append(self._discount_rewards(r))

        return discounted_ep_rs

    def save(self, path, gStep):
        self.saver.save(self.sess, path, global_step=gStep)

    def restore(self, path):
        self.saver.restore(self.sess, path)

