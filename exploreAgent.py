import gym
import gym_Explore2D
import numpy as np
from collections import deque 
from collections import namedtuple
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T  

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"
Experience = namedtuple('Experience',
                        ('state', 'action', 'nextState', 'reward'))


## Hyper Parameters ##
BatchSize = 1000
Gamma = 0.6
# eps_start = 1
# eps_end = 0.01
# eps_decay = 0.001
TargetUpdate = 10
ReplayBufSize = 1000000
LearningRate = 0.001
NumOfEpisodes = 100000

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
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda"
    @staticmethod
    def get_current(policy_net, states, actions):
        max = torch.max(policy_net(states), dim=2)[0][:,0].to(device)
        return max.to(device) 
        #print(max)
        #return max
        #print(policy_net(states).max(dim=2)[0][:,0])
        #return policy_net(states).max(dim=2)[0][:,0]#.gather(dim=1, index=actions.unsqueeze(-1))


    @staticmethod        
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1) \
            .max(dim=1)[0].eq(9).type(torch.bool).to(device)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).float().to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=2)[0][:,0]
        return values.to(device)

def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))
    t1 = torch.stack(batch.state).to(device)
    t2 = torch.stack(batch.action).to(device)
    t3 = torch.stack(batch.reward).to(device)
    t4 = torch.stack(batch.nextState).to(device)
    return (t1,t2,t3,t4)

class Agent():
    def __init__(self,numActions, device):

        self.numActions = numActions
        self.device = device 
    
    def select_action(self, state, network):
        epsilon = 0.1
        if epsilon > random.random():
            #choose to explore
            action = random.randint(0, 3)
            return torch.tensor([action]).to(self.device)
        else:
            #choose to exploit 
            with torch.no_grad():
                return torch.argmax(network.forward(state)).item()
                #return network.forward(state).argmax(dim=1).item()
    


class DQN(nn.Module):
    def __init__(self, mapHeight, mapWidth):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features=((mapHeight)*mapWidth), out_features=256)   
        self.fc2 = nn.Linear(in_features=256, out_features=256)
        self.out = nn.Linear(in_features=256, out_features=4)

    def forward(self, agentState):
        #print(agentState.shape) #10. 1, 65, 64
        layerOutput = torch.flatten(agentState, start_dim=-2, end_dim=-1).float()
        
        #print(layerOutput.shape)
        #print(self.fc1)
        layerOutput = F.relu(self.fc1(layerOutput))
        layerOutput = F.relu(self.fc2(layerOutput))
        layerOutput = self.out(layerOutput)
        return layerOutput

env = gym.make('Explore2D-v0')

agent = Agent(env.numActionsAvailable(), device)
memory = ReplayMemory(ReplayBufSize)

policy_net = DQN(env.getEnvSize()[0],env.getEnvSize()[1]).to(device)
target_net = DQN(env.getEnvSize()[0],env.getEnvSize()[1]).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(params=policy_net.parameters(), lr=LearningRate)

env.setStepLimit(300) #max episode length

totalReward = 0

for episode in range(NumOfEpisodes):
    state = env.reset()
    done = False
    i = 0 
    while not done:
        i += 1
        state = state.unsqueeze(0).unsqueeze(0)

        action = agent.select_action(state, policy_net) + 1
        #print(action)
        nextState, reward, done, info = env.step(action)

        memory.push(state, torch.tensor([action]), nextState, torch.tensor([reward]))

        state = nextState
        
        if memory.sampleAvailable(BatchSize):
            experiences = memory.sample(BatchSize)
            states, actions, rewards, nextStates = extract_tensors(experiences)
            current_q_values = QValues.get_current(policy_net, states.squeeze(1), actions)
            next_q_values = QValues.get_next(target_net, nextStates.unsqueeze(1))
            target_q_values = (next_q_values * Gamma) + rewards
            loss = F.mse_loss(current_q_values, target_q_values[:,0])
            #print(rewards)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if episode % TargetUpdate == 0:
            target_net.load_state_dict(policy_net.state_dict())
    print("Episode finished after {} timesteps".format(i+1))


state = env.reset()
done = False
i = 0
while not done:

    state = state.unsqueeze(0).unsqueeze(0)
    #print('state for greed action',state.shape)
    action = agent.select_action(state, policy_net) + 1
    nextState, reward, done, info = env.step(action)
    state = nextState
    i += 1
    env.render()
print("Episode finished after {} timesteps".format(i+1))

#   randomMove = np.random.randint(low =1, high = 5) #RL agent takes observation and selects a move. RNG in placeholder of agent 
#   observation, reward, done, info = env.step(randomMove) 
#   totalReward += reward
#   if done:
#     print("{} steps remaining".format(observation[1]))
#     print("Episode finished after {} timesteps".format(i+1))
#     print("total reward for episode: " + str(totalReward))
#     break
#env.reset()
