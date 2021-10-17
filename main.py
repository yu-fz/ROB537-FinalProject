import gym
import gym_Explore2D
import wandb

import argparse
import sys

from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
from wandb.integration.sb3 import WandbCallback


if __name__ == "__main__":
      parser = argparse.ArgumentParser()

      """Environment"""

      parser.add_argument("--env_difficulty", default="easy", type=str) #easy, medium, or hard 
      parser.add_argument("--number_of_environments", default=56, type = int)
      parser.add_argument("--max_moves", default = 56, type = int)

      """ RL"""
      parser.add_argument("--total_timesteps", default = 10000000, type=int)


      """logger"""
      parser.add_argument("--logdir",   default="./logs/", type=str)
      parser.add_argument("--save_freq", default = 2000000, type = int)
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

      if args.env_difficulty == "easy":
            
            envName = "Explore2D-Easy-v0"
      
      elif args.env_difficulty == "medium":
            
            envName = "Explore2D-Medium-v0"
      
      elif args.env_difficulty == "hard":
            
            envName = "Explore2D-Hard-v0"

      if mode == 'ppo':
            run = wandb.init(    
            project="sb3",      
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics    

            )

            env = make_vec_env(envName, n_envs = args.number_of_environments)
            #env.setStepLimit(args.max_moves)
            model = PPO("MlpPolicy", env, n_steps = args.max_moves, verbose=1, tensorboard_log="./explorationAgent_tensorboard/", seed = args.seed)
            model.learn(
                  
                  total_timesteps=args.total_timesteps, 
                  tb_log_name= args.run_name, 
                  callback=WandbCallback(
                        model_save_freq = args.save_freq,         
                        model_save_path=".logs/{a}/{b}_{c}".format(a = mode, 
                                                                  b = args.run_name, 
                                                                  c = args.env_difficulty),        
                        verbose=2, 
                        ) 
                  )
            run.finish()


      elif mode == 'a2c':
            run = wandb.init(    
            project="sb3",      
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics    

            )

            env = make_vec_env(envName, n_envs = args.number_of_environments)
            #env.setStepLimit(args.max_moves)
            model = A2C("MlpPolicy", env, n_steps = args.max_moves, verbose=1, tensorboard_log="./explorationAgent_tensorboard/", seed = args.seed)
            model.learn(
                  
                  total_timesteps=args.total_timesteps, 
                  tb_log_name= args.run_name, 
                  callback=WandbCallback(
                        model_save_freq = args.save_freq,         
                        model_save_path=".logs/{a}/{b}_{c}".format(a = mode, 
                                                                  b = args.run_name, 
                                                                  c = args.env_difficulty),         
                        verbose=2, 
                        ) 
                  )
            run.finish()

      elif mode == 'eval':
            env = gym.make(envName)
            model = PPO.load(args.model_path,env)
            model = A2C.load(args.model_path,env)
            stepCounter = 0
            while not done:
                  action = model.predict(obs, deterministic=True)
                  #print(action)
                  obs, reward, done, info = env.step(action)
                  stepCounter+=1
                  print(reward)
            
            env.render()
            print("agent completed {a} time steps".format(a = stepCounter))



