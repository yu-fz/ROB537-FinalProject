import gym
import gym_Explore2D
import numpy as np

#env = gym.make('gym_Explore2D:Explore2D-v0')
env = gym.make('Explore2D-v0')
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
