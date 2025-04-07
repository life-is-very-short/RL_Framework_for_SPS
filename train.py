import os 
import ale_py
import argparse
import torch
import time
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from gymnasium import spaces

from alg.ppo import PPO
from alg.grpo import GRPO
from alg.inter_potential import PARS
from rl_utils import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor_lr", type=  float, default = 1e-3, 
                        help = "actor learning rate")
    parser.add_argument("--critic_lr", type = float, default = 1e-2, 
                        help = "critic learning rate")
    parser.add_argument("--num_episodes", type = int, default = 5000, 
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
    parser.add_argument("--num_steps", type = int, default = 2, 
                        help = "number of steps")
    parser.add_argument("--algo", type = str, default = "ppo", 
                        help = "choose ppo or grpo")
    parser.add_argument("--train_mode", type = int, default = 1, 
                        help = "是否训练模式")
    parser.add_argument("--render_mode", type = str, default = "human", 
                        help = "env中的render模式，human为demo展示，rgb_array为生成gif")
       
    args = parser.parse_args()
    return args

def main(args):
    env = gym.make_vec(args.env_name, args.num_envs)
    torch.manual_seed(234)
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

    potential_agent = PARS(
        env, state_dim, args.hidden_dim, args.critic_lr,
        args.lmbda, args.epochs, args.eps, args.gamma, 
        args.num_steps, args.device
        )

    if args.algo == "ppo":  # PPO
        agent = PPO(
            env, state_dim, args.hidden_dim, action_dim, 
            args.actor_lr, args.critic_lr, args.lmbda, args.epochs, 
            args.eps, args.gamma, args.num_steps, args.device
            )

    elif args.algo == "grpo": # GRPO      
        agent = GRPO(
            env, state_dim, args.hidden_dim, action_dim, 
            args.actor_lr, args.lmbda, args.epochs, 
            args.eps, args.gamma, args.num_steps, args.device
            )
    
    return_list = utils.train_potential_agent(env, agent, potential_agent, args.num_episodes)
    agent.save_model("{}_model".format(args.algo), args.env_name)
    potential_agent.save_model("{}_potential_model".format(args.algo), args.env_name)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('{} on {}'.format(args.algo, args.env_name))
    plt.show()

def test(args):
    env = gym.make_vec(args.env_name, args.num_envs, render_mode = args.render_mode)
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

    if args.algo == "ppo":  # PPO
        agent = PPO(
            env, state_dim, args.hidden_dim, action_dim, 
            args.actor_lr, args.critic_lr, args.lmbda, args.epochs, 
            args.eps, args.gamma, args.num_steps, args.device
            )

    elif args.algo == "grpo": # GRPO      
        agent = GRPO(
            env, state_dim, args.hidden_dim, action_dim, 
            args.actor_lr, args.lmbda, args.epochs, 
            args.eps, args.gamma, args.num_steps, args.device
            )
        
    agent.load_model("{}_model".format(args.algo), args.env_name)

    frames = []
    if args.render_mode == "rgb_array":
        for i_episode in range(2):
            state, _ = env.reset()
            for _ in range(150):
                frames.append(env.render())
                action = agent.take_action(state)
                next_state, reward, done, _, _ = env.step(action)
                state = next_state
        env.close()
        utils.display_frames_as_gif(frames, args.env_name, args.algo)
    elif args.render_mode == "human":
        while True:
            state, _ = env.reset()
            for _ in range(500):
                env.render()
                action = agent.take_action(state)
                next_state, reward, done, _, _ = env.step(action)
                state = next_state
                
            time.sleep(1)


if __name__ == "__main__":
    args = parse_args()
    if args.train_mode == True:
        main(args)
    test(args)
            
    