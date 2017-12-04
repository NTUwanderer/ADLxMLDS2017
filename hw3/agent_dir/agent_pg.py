from agent_dir.agent import Agent
from agent_dir.RL_brain3 import PolicyGradient
import numpy as np
from scipy.misc import imresize

def kill_background_grayscale(image, bg):
    H, W, _ = image.shape

    R = image[..., 0]
    G = image[..., 1]
    B = image[..., 2]

    cond = (R == bg[0]) & (G == bg[1]) & (B == bg[2])

    image = np.zeros((H, W))
    image[~cond] = 1

    return image

def resize_image(image, new_HW):
    return imresize(image, new_HW, interp='nearest')

def crop_ROI(image, height_range=(35, 193), width_range=(0, 160)):
    h_beg, h_end = height_range
    w_beg, w_end = width_range
    return image[h_beg:h_end, w_beg:w_end, ...]

def pipeline(image):
    image = crop_ROI(image, (35, 193))
    image = resize_image(image, (80, 80))
    image = kill_background_grayscale(image, (144, 72, 17))
    image = np.expand_dims(image, axis=2)

    return image

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        self.env = env
        self.env.seed(1)

        self.RL = PolicyGradient(
            n_actions=self.env.get_action_space().n,
            n_features=[80, 80, 1],
            learning_rate=0.02,
            reward_decay=0.99,
        )
        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            self.RL.restore('models/model_pg-99')

        else:
            self.RL.restore('models/model_pg-99')

        ##################
        # YOUR CODE HERE #
        ##################


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################



    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################

        batch_size = 1

        for i_episode in range(3000):
        
            for i in range(batch_size):

                self.RL.add_episode()

                observation = self.env.reset()
                observation = pipeline(observation)
                observation_ = observation

                while True:
        
                    actions, values = self.RL.get_actions_values([observation_])
                    action = actions[0]
                    value = values[0]
                    observation = observation_

                    observation_, reward, done, info = self.env.step(action)
                    observation_ = pipeline(observation_)
        
                    self.RL.store_transition(observation, action, reward, value)
        
                    if done:
                        break
        
            ep_rs_sum = 0.0
            for i in range(batch_size):
                ep_rs_sum += sum(self.RL.ep_rs[i])

            ep_rs_sum /= batch_size
        
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01

            print("episode:", i_episode, "  reward:", int(running_reward), ",  current reward: ", ep_rs_sum)
            print("actions: ", self.RL.checkActDist())
        
            self.RL.learn()

            if (i_episode + 1) % 30 == 0:
                self.RL.save('models/model2_pg', int((i_episode + 1) / 30 - 1))

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
        observation = pipeline(observation)
        action = self.RL.get_actions([observation])[0]
        return action

