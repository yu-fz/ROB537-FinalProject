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
            
            stepCounter = 0
            done = False
            obs = env.reset()
            #print(obs)
            rewardCnt = 0 
            while not done:
                  action, _states = model.predict(obs, deterministic=True)
                  print(action)
                  obs, reward, done, info = env.step(action)
                  stepCounter+=1
                  rewardCnt += reward
                  #print(obs)
                  #print(reward)
            
            env.render()
            print("total reward : {rewardCnt}".format(rewardCnt = rewardCnt))

            print("agent completed {a} time steps".format(a = stepCounter))



