from agent_dir.agent import Agent
from agent_dir.RL_brain import PolicyGradient
import numpy as np

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        if args.test_pg:
            #you can load your model here
            print('loading trained model')

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
        if not hasattr(self, 'RL'):
        
            self.RL = PolicyGradient(
                n_actions=self.env.get_action_space().n,
                n_features=self.env.get_observation_space().shape,
                learning_rate=0.02,
                reward_decay=0.99,
            )

            self.RL.restore('models/model_pg-0')


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.RL = PolicyGradient(
            n_actions=self.env.get_action_space().n,
            n_features=self.env.get_observation_space().shape,
            learning_rate=0.02,
            reward_decay=0.99,
        )

        for i_episode in range(3000):
        
            observation = np.reshape(self.env.reset(), [-1])
        
            while True:
        
                action = self.RL.choose_action(observation)
        
                observation_, reward, done, info = self.env.step(action)
        
                self.RL.store_transition(observation, action, reward)
        
                if done:
                    ep_rs_sum = sum(self.RL.ep_rs)
        
                    if 'running_reward' not in globals():
                        running_reward = ep_rs_sum
                    else:
                        running_reward = running_reward * 0.99 + ep_rs_sum * 0.01

                    print("episode:", i_episode, "  reward:", int(running_reward))
        
                    vt = self.RL.learn()
        
                    break
        
                observation = np.reshape(observation_, [-1])

            if (i_episode + 1) % 30 == 0:
                self.RL.save('models/model_pg', int((i_episode + 1) / 30 - 1))

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
        return self.env.get_random_action()

