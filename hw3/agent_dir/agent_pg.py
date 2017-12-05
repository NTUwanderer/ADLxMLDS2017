from agent_dir.agent import Agent
from agent_dir.RL_brain4 import PolicyGradient
import numpy as np

D = 80 * 80

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

        print ("init")

        self.RL = PolicyGradient(
            n_actions=self.env.get_action_space().n,
            n_features=D,
            learning_rate=0.02,
            reward_decay=0.99,
        )

        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            self.RL.restore('models/model_pg-0')

        ##################
        # YOUR CODE HERE #
        ##################

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

        for i_episode in range(3000):
        
            prev_x = None
            observation = self.env.reset()
        
            while True:

                cur_x = prepro(observation)

                x = cur_x - prev_x if prev_x is not None else np.zeros(D)
                prev_x = cur_x
        
                aprob = self.RL.choose_action(x)
                action = 2 if np.random.uniform() < aprob else 3 # Choose between 2 and 3
                y = 0 if action == 2 else 1 # fake label

                observation, reward, done, info = self.env.step(action)
        
                self.RL.store_transition(x, y, reward)
        
                if done:
                    ep_rs_sum = sum(self.RL.ep_rs)
        
                    if 'running_reward' not in globals():
                        running_reward = ep_rs_sum
                    else:
                        running_reward = running_reward * 0.99 + ep_rs_sum * 0.01

                    print("episode:", i_episode, "  reward:", int(running_reward))
                    print("actions: ", self.RL.checkActDist())
        
                    vt = self.RL.learn()
        
                    break
        

            if (i_episode + 1) % 30 == 0:
                self.RL.save('models/model4_pg', int((i_episode + 1) / 30 - 1))

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
        cur_x = prepro(observation)
        x = cur_x - self.prev_state if self.prev_state is not None else np.zeros(D)
        self.prev_state = cur_x

        aprob = self.RL.choose_action(x)
        action = 2 if np.random.uniform() < aprob else 3 # Choose between 2 and 3

        return action

