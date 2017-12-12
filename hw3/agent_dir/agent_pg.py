from agent_dir.agent import Agent
import numpy as np
import tensorflow as tf

D = 80 * 80
batch_size = 16

n_obs = 80 * 80           # dimensionality of observations
h = 200                   # number of hidden layer neurons
n_actions = 3             # number of available actions
learning_rate = 1e-3
gamma = .99               # discount factor for reward
decay = 0.99              # decay rate for RMSProp gradients
save_path='pg_models/pong.ckpt'


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        """
        self.RL = PolicyGradient(
            #n_actions=self.env.get_action_space().n,
            n_actions=3,
            n_features=D * 2,
            learning_rate=0.02,
            reward_decay=0.99,
            batch_size=batch_size,
        )

        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            self.RL.restore('models/model_pg-0')
        """
        ##################
        # YOUR CODE HERE #
        ##################
        tf_model = {}
        with tf.variable_scope('layer_one',reuse=False):
            xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(n_obs), dtype=tf.float32)
            tf_model['W1'] = tf.get_variable("W1", [n_obs * 2, h], initializer=xavier_l1)
        with tf.variable_scope('layer_two',reuse=False):
            xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(h), dtype=tf.float32)
            tf_model['W2'] = tf.get_variable("W2", [h,n_actions], initializer=xavier_l2)
        
        def tf_policy_forward(x): #x ~ [1,D]
            h = tf.matmul(x, tf_model['W1'])
            h = tf.nn.relu(h)
            logp = tf.matmul(h, tf_model['W2'])
            p = tf.nn.softmax(logp)
            return p
        
        # downsampling
        def prepro(I):
            """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
            I = I[35:195] # crop
            I = I[::2,::2,0] # downsample by factor of 2
            I[I == 144] = 0  # erase background (background type 1)
            I[I == 109] = 0  # erase background (background type 2)
            I[I != 0] = 1    # everything else (paddles, ball) just set to 1
            return I.astype(np.float).ravel()
        
        # tf placeholders
        tf_x = tf.placeholder(dtype=tf.float32, shape=[None, n_obs * 2],name="tf_x")
        tf_y = tf.placeholder(dtype=tf.float32, shape=[None, n_actions],name="tf_y")
        tf_epr = tf.placeholder(dtype=tf.float32, shape=[None,1], name="tf_epr")
        
        # tf optimizer op
        tf_aprob = tf_policy_forward(tf_x)
        loss = tf.nn.l2_loss(tf_y-tf_aprob)
        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay)
        tf_grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables(), grad_loss=tf_epr)
        train_op = optimizer.apply_gradients(tf_grads)
        
        # tf graph initialization
        sess = tf.InteractiveSession()
        tf.initialize_all_variables().run()
        
        # try load saved model
        self.saver = tf.train.Saver(tf.all_variables())
        load_was_success = True # yes, I'm being optimistic
        try:
            save_dir = '/'.join(save_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(save_dir)
            load_path = ckpt.model_checkpoint_path
            self.saver.restore(sess, load_path)
        except:
            print ("no saved model to load. starting new session")
            load_was_success = False
        else:
            print ("loaded model: {}".format(load_path))
            self.saver = tf.train.Saver(tf.all_variables())
            episode_number = int(load_path.split('-')[-1])
        
        self.sess = sess
        self.train_op = train_op
        self.tf_x = tf_x
        self.tf_y = tf_y
        self.tf_epr = tf_epr
        self.tf_aprob = tf_aprob

        self.env = env
        self.env.seed(1)


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.prev_state = None
        


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        env = self.env
        sess = self.sess
        train_op = self.train_op
        tf_x = self.tf_x
        tf_y = self.tf_y
        tf_epr = self.tf_epr
        tf_aprob = self.tf_aprob

        # gamespace 
        observation = env.reset()
        prev_x = None
        xs,rs,ys = [],[],[]
        running_reward = None
        reward_sum = 0
        episode_number = 0

        record_rewards = []

        record_file = open('pg_record', 'w', 1000)
        
        # training loop
        while True:
        #     if True: env.render()
        
            # preprocess the observation, set input to network to be difference image
            cur_x = prepro(observation)
            x = cur_x - prev_x if prev_x is not None else np.zeros(n_obs)
            prev_x = cur_x
        
            # stochastically sample a policy from the network
            feed = {tf_x: np.reshape(np.stack((x, cur_x)), (1,-1))}
            aprob = sess.run(tf_aprob,feed) ; aprob = aprob[0,:]
            action = np.random.choice(n_actions, p=aprob)
            label = np.zeros_like(aprob) ; label[action] = 1
        
            # step the environment and get new measurements
            observation, reward, done, info = env.step(action+1)
            reward_sum += reward
            
            # record game history
            xs.append(np.reshape(np.stack((x, cur_x)), (1, -1))) ; ys.append(label) ; rs.append(reward)
            
            if done:
                # update running reward
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                if len(record_rewards) == 30:
                    record_rewards = record_rewards[1:]

                record_rewards.append(reward_sum)
                
                temp_rs = np.vstack(rs)
                discounted = np.zeros_like(temp_rs)
                running_add = 0
                for t in reversed(range(0, len(temp_rs))):
                    if temp_rs[t] != 0.0:
                        running_add = 0

                    running_add = running_add * gamma + temp_rs[t]
                    discounted[t] = running_add

                discounted -= np.mean(discounted)
                discounted /= np.std(discounted)

                # parameter update
                feed = {tf_x: np.vstack(xs), tf_epr: np.vstack(discounted), tf_y: np.vstack(ys)}
                _ = sess.run(train_op,feed)
                
                # print progress console
                if episode_number % 10 == 0:
                    mean_r = sum(record_rewards) / float(len(record_rewards))
                    print ('ep {}: reward: {}, mean reward: {:3f}'.format(episode_number, reward_sum, mean_r))
                    record_file.write('{}, {:3f}\n'.format(episode_number, mean_r))
                    #print ('ep {}: reward: {}, mean reward: {:3f}'.format(episode_number, reward_sum, running_reward))
                else:
                    print ('\tep {}: reward: {}'.format(episode_number, reward_sum))
                
                # bookkeeping
                xs,rs,ys = [],[],[] # reset game history
                episode_number += 1 # the Next Episode
                observation = env.reset() # reset env
                reward_sum = 0
                if episode_number % 50 == 0:
                    self.saver.save(sess, save_path, global_step=episode_number)
                    print ("SAVED MODEL #{}".format(episode_number))

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        sess = self.sess
        tf_aprob = self.tf_aprob
        tf_x = self.tf_x

        cur_x = prepro(observation)
        x = cur_x - self.prev_state if self.prev_state is not None else np.zeros(n_obs)
        self.prev_state = cur_x

        feed = {tf_x: np.reshape(np.stack((x, cur_x)), (1,-1))}
        aprob = sess.run(tf_aprob,feed) ; aprob = aprob[0,:]
        action = np.random.choice(n_actions, p=aprob)

        return action + 1

