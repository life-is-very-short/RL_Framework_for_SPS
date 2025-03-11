import os 
import ale_py
import argparse
import torch
import gymnasium as gym

from gymnasium import spaces

from alg.ppo import PPO
from alg.grpo import GRPO
from rl_utils import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor_lr", type=  float, default = 1e-3, 
                        help = "actor learning rate")
    parser.add_argument("--critic_lr", type = float, default = 1e-2, 
                        help = "critic learning rate")
    parser.add_argument("--num_episodes", type = int, default = 500, 
                        help = "number of episodes")
    parser.add_argument("--hidden_dim", type = int, default = 128, 
                        help = "hidden layer dimension")
    parser.add_argument("--gamma", type = float, default = 0.98, 
                        help = "discount factor")
    parser.add_argument("--lmbda", type = float, default = 0.95, 
                        help = "gae parameter")
    parser.add_argument("--epochs", type = int, default = 10, 
                        help = "number of epochs")
    parser.add_argument("--eps", type = float, default = 0.2, 
                        help = "ppo clip range")
    parser.add_argument("--device", type = str, default = "cuda" if torch.cuda.is_available() else "cpu", 
                        help = "device")
    parser.add_argument("--env_name", type = str, default = "CartPole-v1", 
                        help = "environment name")
    parser.add_argument("--num_envs", type = int, default = 4, 
                        help = "number of environments")
    parser.add_argument("--num_steps", type = int, default = 512, 
                        help = "number of steps")
    parser.add_argument("--algo_choice", type = int, default = 0, 
                        help = "choose ppo or grpo")
       
    args = parser.parse_args()
    return args

def main(args):
    env = gym.make_vec(args.env_name, args.num_envs)
    torch.manual_seed(0)
    if isinstance(env.action_space, (spaces.Discrete, spaces.MultiDiscrete)):
        action_dim = env.action_space[0].n
    else:
        try:
            action_dim = env.action_space.shape[1]
        except:
            action_dim = 1
    
    try:
        state_dim = env.observation_space.shape[1]
    except:
        state_dim = 1

    if args.algo_choice == 0:  # PPO
        agent = PPO(env, state_dim, args.hidden_dim, action_dim, args.actor_lr, args.critic_lr, 
                args.lmbda, args.epochs, args.eps, args.gamma, args.num_steps, args.device)

    elif args.algo_choice == 1: # GRPO      
        agent = GRPO(state_dim, args.hidden_dim, action_dim, args.actor_lr, 
                args.lmbda, args.epochs, args.eps, args.gamma, args.num_steps, args.device)
        
    return_list = utils.train_on_policy_agent(env, agent, args.num_episodes)
    agent.save_model("ppo_model", args.env_name)

if __name__ == "__main__":
    args = parse_args()
    main(args)
    