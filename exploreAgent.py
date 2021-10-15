import gym
import gym_Explore2D
import numpy as np
from collections import deque 
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T  

env = gym.make('Explore2D-v0')


Experience = namedtuple('Transition',
                        ('state', 'action', 'nextState', 'reward'))


class ReplayMemory():

    def __init__(self, capacity):
        self.memory = deque(maxlen = capacity)

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Experience(*args))

    def sample(self, batchSize):
        return random.sample(self.memory, batchSize)

    def sampleAvailable(self, batchSize):
        return len(self.memory) >= batchSize


class epsilonGreedy():
    ...

class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod        
    def get_next(target_net, next_states):                
        final_state_locations = next_states.flatten(start_dim=1) \
            .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values

class Agent():
    def __init__(self,numActions, device):

        self.numActions = numActions
        self.device = device 
    
    def select_action(self, state, network):
        epsilon = 0.05
        if epsilon > random.random():
            #choose to explore
            action = random.randint(1, self.numActions)
            return torch.tensor([action]).to(self.device)
        else:
            #choose to exploit 
            with torch.no_grad():
                return network(state).argmax(dim=1).to(self.device)
    


class DQN(nn.Module):
    def __init__(self, mapHeight, mapWidth):
        super().__init__()

        self.fc1 = nn.Linear(in_features=mapHeight*mapWidth, out_features=64)   
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.out = nn.Linear(in_features=64, out_features=4)

    def forward(self, agentState):
        layerOutput = agentState.flatten(start_dim=1)
        layerOutput = F.relu(self.fc1(layerOutput))
        layerOutput = F.relu(self.fc2(layerOutput))
        layerOutput = self.out(layerOutput)
        return layerOutput




env.reset()
totalReward = 0
for i in range(10000):
  randomMove = np.random.randint(low =1, high = 5) #RL agent takes observation and selects a move. RNG in placeholder of agent 
  observation, reward, done, info = env.step(randomMove) 
  totalReward += reward
  if done:
    print("Episode finished after {} timesteps".format(i+1))
    print("total reward for episode: " + str(totalReward))
    break
#env.reset()
