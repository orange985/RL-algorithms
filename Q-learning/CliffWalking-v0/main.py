import gymnasium as gym
import numpy as np
import yaml
import matplotlib.pyplot as plt
from gymnasium import Env

# read config file
def get_config(file_name: str)->dict:
    with open(file_name, 'r') as file:
        config = yaml.safe_load(file)
    return config

# draw the curve of reward
def draw(reward_list: list):
    plt.plot(reward_list)
    plt.xlabel('epoch')
    plt.ylabel('reward')
    plt.show()


class Agent:
    def __init__(self, config: dict, num_states: int, num_actions: int):
        self.epoch=config['epoch']
        self.epsilon=config['epsilon']
        self.alpha=config['alpha']
        self.gamma=config['gamma']
        
        self.Q_table=np.zeros((num_states,num_actions))         #initialize Q table
        
    # epsilon-greedy policy to choose action
    def choose_action(self, state: int)-> int:
        if np.random.uniform()<=self.epsilon:
            action=env.action_space.sample()
        else:
            action=np.argmax(self.Q_table[state,:])
            
        return action

    # update Q table
    def update(self, state: int, action: int, reward: int, next_state: int, terminated: bool):
        if terminated:
            Q_target=reward
        else:
            Q_target=reward+self.gamma*np.max(self.Q_table[next_state,:])
            
        self.Q_table[state,action]+=self.alpha*(Q_target-self.Q_table[state,action])
        
    # train the agent
    def train(self, env: Env)-> list:
        reward_list=[]
        for epoch in range(self.epoch):
            state=env.reset()[0]
            epoch_reward=0
            while True:
                action=self.choose_action(state)
                next_state,reward,terminated,truncated,info=env.step(action)
                self.update(state,action,reward,next_state,terminated)
                state=next_state
                epoch_reward+=reward
                
                if terminated or truncated:
                    break
                
            reward_list.append(epoch_reward)
            self.epsilon*=0.99      # decreasing epsilon
            
            print(epoch,epoch_reward)
            
        return reward_list
        
        
if __name__=='__main__':
    env=gym.make("CliffWalking-v0")
    num_states=env.observation_space.n  # num_states=48
    num_actions=env.action_space.n      # num_actions=4
    config=get_config("./config.yaml")
    agent=Agent(config,num_states,num_actions)
    reward_list=agent.train(env)
    draw(reward_list)
    env.close()
