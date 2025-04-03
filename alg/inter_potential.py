import numpy as np
import torch
import torch.nn as nn

from gymnasium import spaces
from torch.nn import functional as F

from rl_utils.nn_model import PolicyNet_Discrete, ValueNet, PolicyNet_Continue
from rl_utils import utils


class PPO:
    ''' Potential-based Auto-Reward Shaping '''
    def __init__(self, env, state_dim, hidden_dim, critic_lr,
                 lmbda, epochs, eps, gamma, num_steps, device):
        '''
        :param state_dim: 状态空间维度
        :param hidden_dim: 隐藏层维度
        :param action_dim: 动作空间维度
        :param actor_lr: actor学习率
        :param critic_lr: critic学习率
        :param lmbda: GAE参数
        :param epochs: 一条序列的数据用来训练轮数
        :param eps: PPO中截断范围的参数
        :param gamma: 折扣因子
        :param device: 训练设备
        '''
        self.env = env
        self.potential = ValueNet(state_dim, hidden_dim).to(device)
        self.potential_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.num_steps = num_steps
        self.device = device
        
    def update(self, transition_dict): 
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions'])).to(self.device)
        if isinstance(self.env.action_space, (spaces.Discrete, spaces.MultiDiscrete)):
            actions = actions.unsqueeze(dim=-1)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).unsqueeze(dim=-1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        termination = torch.tensor(np.array(transition_dict['termination']), dtype=torch.float).unsqueeze(dim=-1).to(self.device)
        trucation = torch.tensor(np.array(transition_dict['trucation']), dtype=torch.float).unsqueeze(dim=-1).to(self.device)
        dones = termination 
        td_target = rewards + self.gamma * self.potential(next_states) * (1 - dones)

        potential_loss = torch.mean(F.mse_loss(self.potential(states), td_target.detach()))  # TD error

        self.potential_optimizer.zero_grad()
        potential_loss.backward()
        self.potential_optimizer.step()
    
    def save_model(self, path, env_name):
        torch.save(self.potential.state_dict(), 'model/{}/potential_{}.pth'.format(path, env_name))

    def load_model(self, path, env_name):
        self.potential.load_state_dict(torch.load('model/{}/potential_{}.pth'.format(path, env_name)))
        self.potential.eval()