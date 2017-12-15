import tensorflow as tf
import numpy as np

class DQN:

    def __init__(self, env, test=False):
        self.n_actions = env.action_space.n
        self.learning_rate = 0.0001
        self.gamma = 0.99
        self.batch_size = 32
        self.memory_size = 10000
        self.memory_count = 0
        self.learn_step_count = 0
        self.update_target_freq = 1000
        if test:
            self.epsilon = 0
            self.epsilon_decay = 0
            self.epsilon_min = 0
        else:
            self.epsilon = 1
            self.epsilon_decay = 1e-6
            self.epsilon_min = 0.05

        self.build_net()

        self.reset_memory()

        self.sess  = tf.Session()
        self.saver = tf.train.Saver()

        q_params = tf.get_collection('Q_network')
        t_params = tf.get_collection('target_network')
        self.update_target_op = [tf.assign(t, q) for t, q in zip(t_params, q_params)]

        self.sess.run(tf.global_variables_initializer())
        self.loss_record = []

    def build_net(self):

        def conv2d_layer(input_tensor, filter_shape, strides, name, collections, padding='SAME', activation=tf.nn.relu):
            with tf.variable_scope(name):
                filters = tf.get_variable('filters', filter_shape, dtype=tf.float32, collections=collections)
                bias = tf.get_variable('bias', [filter_shape[3]], dtype=tf.float32, collections=collections)
                conv = tf.nn.conv2d(input_tensor, filters, strides, padding)
                output = activation(tf.nn.bias_add(conv, bias))
                return output
        
        def dense_layer(input_tensor, input_dim, output_dim, name, collections, activation=tf.nn.relu):
            with tf.variable_scope(name):
                W = tf.get_variable('weights', [input_dim, output_dim], dtype=tf.float32, collections=collections)
                b = tf.get_variable('bias', [output_dim], dtype=tf.float32, collections=collections)
                output = activation(tf.matmul(input_tensor, W))
                return output

        def lrelu(x, alpha=0.2):
            return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

        ####################### Q Network ##########################
        self.tf_states = tf.placeholder(tf.float32, [None, 84, 84, 4], name='states')

        with tf.variable_scope('Q_network'):
            collection_name = ['Q_network', tf.GraphKeys.GLOBAL_VARIABLES]
            conv1 = conv2d_layer(self.tf_states, [8, 8, 4, 32], [1, 4, 4, 1], 'conv1', collection_name)
            conv2 = conv2d_layer(conv1, [4, 4, 32, 64], [1, 2, 2, 1], 'conv2', collection_name)
            conv3 = conv2d_layer(conv2, [3, 3, 64, 64], [1, 1, 1, 1], 'conv3', collection_name)
            flatten_dim = np.prod(conv3.get_shape().as_list()[1:])
            flatten = tf.reshape(conv3, [-1, flatten_dim])
            dense = dense_layer(flatten, flatten_dim, 512, 'dense', collection_name, lrelu)
            self.Q_value = dense_layer(dense, 512, self.n_actions, 'output', collection_name, tf.identity)

        #################### Target Q Network ######################
        self.tf_states_t = tf.placeholder(tf.float32, [None, 84, 84, 4], name='states_target')

        with tf.variable_scope('target_network'):
            collection_name = ['target_network', tf.GraphKeys.GLOBAL_VARIABLES]
            conv1 = conv2d_layer(self.tf_states_t, [8, 8, 4, 32], [1, 4, 4, 1], 'conv1', collection_name)
            conv2 = conv2d_layer(conv1, [4, 4, 32, 64], [1, 2, 2, 1], 'conv2', collection_name)
            conv3 = conv2d_layer(conv2, [3, 3, 64, 64], [1, 1, 1, 1], 'conv3', collection_name)
            flatten_dim = np.prod(conv3.get_shape().as_list()[1:])
            flatten = tf.reshape(conv3, [-1, flatten_dim])
            dense = dense_layer(flatten, flatten_dim, 512, 'dense', collection_name, lrelu)
            self.Q_target = dense_layer(dense, 512, self.n_actions, 'output', collection_name, tf.identity)

        ################### Loss and Optimizer #####################
        self.tf_actions = tf.placeholder(tf.int32, [None], name='actions')
        self.tf_y = tf.placeholder(tf.float32, [None], name='y')

        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, tf.one_hot(self.tf_actions, self.n_actions)), axis=-1)
        self.loss = tf.reduce_mean(tf.squared_difference(self.tf_y, Q_action))
        self.train_step = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99).minimize(self.loss)

    def learn(self):
        '''
        Update parameters.
        '''
        # Update target network
        if self.learn_step_count % self.update_target_freq == 0:
            self.sess.run(self.update_target_op)

        # Sample batch from memory
        if self.memory_count > self.memory_size:
            indices = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            indices = np.random.choice(self.memory_count, size=self.batch_size)
        state_batch = self.states[indices]
        action_batch = self.actions[indices]
        reward_batch = self.rewards[indices]
        next_state_batch = self.next_states[indices]
        done_batch = self.done[indices]

        # Calculate Target Q values
        Q_target = self.sess.run(self.Q_target, feed_dict={self.tf_states_t: next_state_batch})

        # Calculate y
        y_batch = [reward_batch[i] if done_batch[i] else reward_batch[i] + self.gamma * np.max(Q_target[i]) 
            for i in range(self.batch_size)]

        # Update parameters
        _, loss = self.sess.run([self.train_step, self.loss], feed_dict={self.tf_states: state_batch,
                                                                         self.tf_actions: action_batch,
                                                                         self.tf_y: y_batch})
        self.loss_record.append(loss)
        self.learn_step_count += 1

    def reset_memory(self):
        '''
        Reset the transition memories.
        '''
        self.states = np.zeros((self.memory_size, 84, 84, 4))
        self.actions = np.zeros(self.memory_size)
        self.rewards = np.zeros(self.memory_size)
        self.next_states = np.zeros((self.memory_size, 84, 84, 4))
        self.done = np.zeros(self.memory_size)

    def store_step(self, state, action, reward, next_state, done):
        '''
        Store a transition into the memory.
        '''
        index = self.memory_count % self.memory_size
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.done[index] = done
        self.memory_count += 1

    def choose_action(self, state):
        '''
        Choose an action given provided state.
        '''
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            action_values = self.sess.run(self.Q_value, feed_dict={self.tf_states: np.expand_dims(state, 0)})
            action = np.argmax(action_values)
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min
        return action

    def save_model(self, file_path, step):
        '''
        Save current model to file.
        '''
        self.saver.save(self.sess, file_path, global_step=step)

    def load_model(self, model_path):
        '''
        Load pretrained model from file.
        '''
        try:
            save_dir = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(save_dir)
            load_path = ckpt.model_checkpoint_path
            self.saver.restore(self.sess, load_path)
        except:
            print ("no saved model to load. starting new session")
        else:
            print ("loaded model: {}".format(load_path))
            

class DoubleDQN(DQN):

    def __init__(self, env, test=False):
        super(DoubleDQN,self).__init__(env, test)

    def learn(self):
        '''
        Update parameters.
        '''
        # Update target network
        if self.learn_step_count % self.update_target_freq == 0:
            self.sess.run(self.update_target_op)

        # Sample batch from memory
        if self.memory_count > self.memory_size:
            indices = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            indices = np.random.choice(self.memory_count, size=self.batch_size)
        state_batch = self.states[indices]
        action_batch = self.actions[indices]
        reward_batch = self.rewards[indices]
        next_state_batch = self.next_states[indices]
        done_batch = self.done[indices]

        # Calculate Target Q values
        Q_target, Q_value_next = self.sess.run([self.Q_target, self.Q_value], feed_dict={self.tf_states_t: next_state_batch,
                                                                                         self.tf_states: next_state_batch})

        # Calculate y
        y_batch = [reward_batch[i] if done_batch[i] else reward_batch[i] + self.gamma * Q_target[i][np.argmax(Q_value_next[i])] 
            for i in range(self.batch_size)]

        # Update parameters
        _, loss = self.sess.run([self.train_step, self.loss], feed_dict={self.tf_states: state_batch,
                                                                         self.tf_actions: action_batch,
                                                                         self.tf_y: y_batch})
        self.loss_record.append(loss)
        self.learn_step_count += 1

class DuelingDQN(DQN):

    def __init__(self, env, test=False):
        super(DuelingDQN,self).__init__(env, test)

    def build_net(self):

        def conv2d_layer(input_tensor, filter_shape, strides, name, collections, padding='SAME', activation=tf.nn.relu):
            with tf.variable_scope(name):
                filters = tf.get_variable('filters', filter_shape, dtype=tf.float32, collections=collections)
                bias = tf.get_variable('bias', [filter_shape[3]], dtype=tf.float32, collections=collections)
                conv = tf.nn.conv2d(input_tensor, filters, strides, padding)
                output = activation(tf.nn.bias_add(conv, bias))
                return output
        
        def dense_layer(input_tensor, input_dim, output_dim, name, collections, activation=tf.nn.relu):
            with tf.variable_scope(name):
                W = tf.get_variable('weights', [input_dim, output_dim], dtype=tf.float32, collections=collections)
                b = tf.get_variable('bias', [output_dim], dtype=tf.float32, collections=collections)
                output = activation(tf.matmul(input_tensor, W))
                return output

        def lrelu(x, alpha=0.2):
            return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

        ####################### Q Network ##########################
        self.tf_states = tf.placeholder(tf.float32, [None, 84, 84, 4], name='states')

        with tf.variable_scope('Q_network'):
            collection_name = ['Q_network', tf.GraphKeys.GLOBAL_VARIABLES]
            conv1 = conv2d_layer(self.tf_states, [8, 8, 4, 32], [1, 4, 4, 1], 'conv1', collection_name)
            conv2 = conv2d_layer(conv1, [4, 4, 32, 64], [1, 2, 2, 1], 'conv2', collection_name)
            conv3 = conv2d_layer(conv2, [3, 3, 64, 64], [1, 1, 1, 1], 'conv3', collection_name)
            flatten_dim = np.prod(conv3.get_shape().as_list()[1:])
            flatten = tf.reshape(conv3, [-1, flatten_dim])
            dense = dense_layer(flatten, flatten_dim, 512, 'dense', collection_name, lrelu)
            V = dense_layer(dense, 512, 1, 'value', collection_name, tf.identity)
            A = dense_layer(dense, 512, self.n_actions, 'advantage', collection_name, tf.identity)
            self.Q_value = V + (A - tf.reduce_mean(A, axis=1, keep_dims=True))

        #################### Target Q Network ######################
        self.tf_states_t = tf.placeholder(tf.float32, [None, 84, 84, 4], name='states_target')

        with tf.variable_scope('target_network'):
            collection_name = ['target_network', tf.GraphKeys.GLOBAL_VARIABLES]
            conv1 = conv2d_layer(self.tf_states_t, [8, 8, 4, 32], [1, 4, 4, 1], 'conv1', collection_name)
            conv2 = conv2d_layer(conv1, [4, 4, 32, 64], [1, 2, 2, 1], 'conv2', collection_name)
            conv3 = conv2d_layer(conv2, [3, 3, 64, 64], [1, 1, 1, 1], 'conv3', collection_name)
            flatten_dim = np.prod(conv3.get_shape().as_list()[1:])
            flatten = tf.reshape(conv3, [-1, flatten_dim])
            dense = dense_layer(flatten, flatten_dim, 512, 'dense', collection_name, lrelu)
            V = dense_layer(dense, 512, 1, 'value', collection_name, tf.identity)
            A = dense_layer(dense, 512, self.n_actions, 'advantage', collection_name, tf.identity)
            self.Q_target = V + (A - tf.reduce_mean(A, axis=1, keep_dims=True))

        ################### Loss and Optimizer #####################
        self.tf_actions = tf.placeholder(tf.int32, [None], name='actions')
        self.tf_y = tf.placeholder(tf.float32, [None], name='y')

        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, tf.one_hot(self.tf_actions, self.n_actions)), axis=-1)
        self.loss = tf.reduce_mean(tf.squared_difference(self.tf_y, Q_action))
        self.train_step = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99).minimize(self.loss)
