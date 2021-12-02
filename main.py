from os import terminal_size
from numpy.core.defchararray import index
from frontiers import FrontierPointFinder
import gym
import gym_Explore2D
import wandb
import torch as th
import argparse
import sys

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps, EvalCallback
from wandb.integration.sb3 import WandbCallback


if __name__ == "__main__":
      parser = argparse.ArgumentParser()

      """Environment"""

      parser.add_argument("--env_map", default= "./maps/gridWorld_easy.csv", type=str) # path to grid map
      parser.add_argument("--number_of_environments", default=56, type = int)
      parser.add_argument("--max_moves", default = 56, type = int)

      """ RL"""
      parser.add_argument("--total_timesteps", default = 10000000, type=int)


      """logger"""
      parser.add_argument("--logdir",   default="./logs/", type=str)
      parser.add_argument("--save_freq", default = 200000, type = int)
      parser.add_argument("--run_name", default= None, type=str)  

      """Seed"""
      parser.add_argument("--seed", default = 0, type = int)

      """Eval"""

      parser.add_argument("--model_path", type=str)

      mode = sys.argv[1]
      print(mode)
      sys.argv.remove(sys.argv[1])
      args = parser.parse_args()
      print(args.run_name)

      envName = "Explore2D-v0"

      if mode == 'ppo':
            run = wandb.init(    
            project="sb3",      
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics    

            )

            path_to_map = args.env_map
            envkwargs["map"] = path_to_map
            env = make_vec_env(envName, n_envs = args.number_of_environments, env_kwargs= envkwargs)
            #env.setStepLimit(args.max_moves)

            checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path='./logs/{run_name}'.format(run_name = args.run_name),
                                         name_prefix='rl_model')

            model = PPO("MultiInputPolicy", env, n_steps = args.max_moves, verbose=1, tensorboard_log="./explorationAgent_tensorboard/", seed = args.seed)
            model.learn(
                  
                  total_timesteps=args.total_timesteps, 
                  tb_log_name= args.run_name, 
                  callback=checkpoint_callback

                  )
            run.finish()
      elif mode == 'dqn':
            run = wandb.init(    
            project="sb3",      
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics    

            )

            path_to_map = args.env_map
            env = gym.make(envName, map = path_to_map)
            #env = make_vec_env(envName, n_envs = args.number_of_environments, map = path_to_map)

            policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[32, 32])

            checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path='./logs/{run_name}'.format(run_name = args.run_name),
                                         name_prefix='rl_model')

            model = DQN("MultiInputPolicy", env, policy_kwargs=policy_kwargs, learning_starts = 500000, verbose=1, exploration_fraction=0.25, tensorboard_log="./explorationAgent_tensorboard/", seed = args.seed)
            model.learn(
                  
                  total_timesteps=args.total_timesteps, 
                  tb_log_name= args.run_name, 
                  callback=checkpoint_callback

                  )
            run.finish()


      elif mode == 'eval':
            path_to_map = args.env_map
            env = gym.make(envName, map = path_to_map)
            model = DQN.load(args.model_path,env)
            obs = env.reset()
            stepCounter = 0
            done = False
            #obs = env.reset()
            #print(freeCoords)
            #print(obs)
            rewardCnt = 0 
            frontierMap = env.returnFrontierMap()
            frontiers = FrontierPointFinder(frontierMap)


            while (env.returnExplorationProgress() < 0.9):
                  #print(env.returnExplorationProgress())
                  #env.resetObjGrid()
                  frontiers.updateFrontierMap(env.returnFrontierMap())
                  frontierWaypoint = frontiers.returnTargetFrontierPoint()
                  #print(frontierWaypoint)
                  env.setObjectiveCoord(frontierWaypoint)
                  #env.resetObjGrid()
                  obs = env.resetFrontier()
                  #env.resetObjGrid()
                  #print(obs)
                  done = False
                  while not done:
                        
                        action, _states = model.predict(obs, deterministic=True)
                        #print(action)
                        obs, reward, done, info = env.step(action)
                        #frontierMap = env.returnFrontierMap()
                        env.render()
                        #env.saveObsImage(stepCounter)
                        stepCounter+=1
                        
                        #rewardCnt += reward

                        #print(obs)
                        #print(reward)
                  
                  env.resetObjGrid()
                  
            env.render()
            print("total reward : {rewardCnt}".format(rewardCnt = rewardCnt))

            print("agent completed {a} time steps".format(a = stepCounter))


      elif mode == 'evalPlot':
            
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd


            path_to_map = args.env_map
            env = gym.make(envName, map = path_to_map)
            model = DQN.load(args.model_path,env)

            trials = 50

            stepData = np.empty([trials,10])
            stepDataRandom = np.empty([trials,10])

            x_axis = 100*np.arange(0,0.91,0.1)

            for i in range(trials):

                  env.reset()
                  stepCounter = 0
                  done = False
                  #obs = env.reset()
                  #print(freeCoords)
                  #print(obs)
                  rewardCnt = 0 
                  frontierMap = env.returnFrontierMap()
                  frontiers = FrontierPointFinder(frontierMap)

                  stepsTaken = []
                  exploreProgress = []

                  while (env.returnExplorationProgress() < 0.9):
                        frontiers.updateFrontierMap(env.returnFrontierMap())
                        frontierWaypoint = frontiers.returnTargetFrontierPoint()

                        env.setObjectiveCoord(frontierWaypoint)
                        obs = env.resetFrontier()
                        #print(obs)
                        done = False
                        while not done:
                              
                              action, _states = model.predict(obs, deterministic=True)
                              #print(action)
                              obs, reward, done, info = env.step(action)
                              #frontierMap = env.returnFrontierMap()
                              #env.render()
                              #env.saveObsImage(stepCounter)
                              stepCounter+=1                              
                              #rewardCnt += reward

                              #print(obs)
                              #print(reward)
                        
                              currProgress = round(env.returnExplorationProgress(),1)
                              if(currProgress not in exploreProgress):
                                    exploreProgress.append(currProgress)
                                    stepsTaken.append(stepCounter)
                        env.resetObjGrid()

                  stepData[i,:] = stepsTaken

            print(stepData)

            df = pd.DataFrame(data = stepData, columns= x_axis).T

            for row in stepData:

                  sns.regplot(x_axis, row, fit_reg=False, order = 2, color = "darkslateblue")
                  #plt.scatter(x_axis, row, c = "blue")
            #       df.append(row, ignore_index=True)
            #sns.lineplot(data = df, err_style = "band",ci=68)
            #plt.legend(["DQN Agent"])
            # for i in range(trials):
            #       env.reset()
            #       totalReward = 0
            #       stepCounter = 0
            #       stepsTaken = []
            #       exploreProgress = []
            #       while (env.returnExplorationProgress() < 0.9):
            #             randomMove = np.random.randint(low =0, high = 5) #RL agent takes observation and selects a move. RNG in placeholder of agent 
            #             observation, reward, done, info = env.step(randomMove) 
            #             #totalReward += reward
            #             stepCounter+=1
            #             currProgress = round(env.returnExplorationProgress(),1)
            #             if(currProgress not in exploreProgress):
            #                   exploreProgress.append(currProgress)
            #                   stepsTaken.append(stepCounter)
            #       stepDataRandom[i,:] = stepsTaken
            
            # for row in stepDataRandom:

            #       sns.regplot(x_axis, row, fit_reg=False, color = "seagreen", marker = "x")
                  
            #plt.legend(["random agent"])
            #print(df)
            plt.xlabel("percentage of map uncovered")
            plt.ylabel("path cost")
            plt.grid()
            plt.show()
            #env.render()
            # print("total reward : {rewardCnt}".format(rewardCnt = rewardCnt))

            # print("agent completed {a} time steps".format(a = stepCounter))


