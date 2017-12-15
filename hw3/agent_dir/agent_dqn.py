from agent_dir.agent import Agent
from agent_dir.DQNs import DQN, DuelingDQN
from test import test
from environment import Environment
import tensorflow as tf
import numpy as np
import time
import sys

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)
        self.DQN = DuelingDQN(env, args.test_dqn)
        # self.resume = args.resume
        self.resume = False
        self.args = args

        if args.test_dqn:
            self.DQN.load_model('model_dldqn/model.ckpt-3800')
            print('loading trained model')

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
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        total_steps = 10000000
        start_steps = 10000
        step_count = 0
        frame_count = 0
        learn_count = 0
        episode_count = 0
        episode_reward = 0
        learn_freq = 4
        f = open('train_dldqn.csv', 'w')
        if self.resume:
            step_count, frame_count, learn_count, episode_count, episode_reward, self.DQN.memory_count, self.DQN.learn_step_count, self.DQN.epsilon = np.load('model_dqn/record.npy')
            self.DQN.load_model(tf.train.latest_checkpoint('model_dldqn'))
        else:
            print('episode,reward', file=f)
        t_start = time.time()
        t_episode = time.time()

        state = self.env.reset()
        for i in range(total_steps):
            action = self.DQN.choose_action(state)
            next_state, reward, done, info = self.env.step(action)
            self.DQN.store_step(state, action, reward, next_state, done)
            state = next_state
            if (step_count >= start_steps) and (step_count % learn_freq == 0):
                self.DQN.learn()
                learn_count += 1
            
            step_count += 1 
            frame_count += 1
            episode_reward += reward

            if done:
                episode_count += 1
                sys.stdout.write('\rEpisode # %-5d | steps: %-5d | episode reward: %f                    \n' % (
                                    episode_count, frame_count, episode_reward))
                sys.stdout.flush()
                print('%d,%f'%(episode_count,episode_reward), file=f)
                frame_count = 0
                episode_reward = 0
                t_episode = time.time()
                state = self.env.reset()

            t_elapsed = int(time.time() - t_start)
            sys.stdout.write('\r#--- Step: %-7d  epsilon: %-6.4f  elapsed: %02d:%02d:%02d   updated times: %-6d' % (
                step_count, self.DQN.epsilon, t_elapsed // 3600, t_elapsed % 3600 // 60, t_elapsed % 60, learn_count))

            if (step_count > 100000) and (step_count % 100000 == 0):
                self.DQN.save_model('model_dldqn/model.ckpt', int(step_count/1000))
                np.save('model_dldqn/record.npy', [step_count, frame_count, learn_count, episode_count, episode_reward, self.DQN.memory_count, self.DQN.learn_step_count, self.DQN.epsilon])
                if (step_count > 2000000) and (step_count % 500000 == 0):
                    test_env = Environment('BreakoutNoFrameskip-v4', self.args, atari_wrapper=True, test=True)
                    test_score = test(self, test_env, 100)
                    print('################################################################                                   ')
                    print('Step: %d  | testing score: %f' % (step_count, test_score))
                    print('################################################################')
                    if test_score > 40:
                        wait = input("Baseline passed. Continue? [y/n] ")
                        if wait == n: break

        f.close()

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        return self.DQN.choose_action(observation)

