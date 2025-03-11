import numpy as np
import torch
import torch.nn as nn

from gymnasium import spaces
from torch.nn import functional as F

from rl_utils.nn_model import PolicyNet_Discrete, ValueNet, PolicyNet_Continue
from rl_utils import utils


class GRPO:
    ''' GRPO算法,适用于单步决策环境 '''
    def __init__(self, env, state_dim, hidden_dim, action_dim, actor_lr,
                 lmbda, epochs, eps, gamma, num_steps, device):
        '''
        :param state_dim: 状态空间维度
        :param hidden_dim: 隐藏层维度
        :param action_dim: 动作空间维度
        :param actor_lr: actor学习率
        :param lmbda: GAE参数
        :param epochs: 一条序列的数据用来训练轮数
        :param eps: 截断范围的参数
        :param gamma: 折扣因子
        :param device: 训练设备
        '''
        self.env = env
        self.actor = PolicyNet_Discrete(state_dim, hidden_dim, action_dim).to(device) if isinstance(self.env.action_space, 
                    (spaces.Discrete, spaces.MultiDiscrete)) else PolicyNet_Continue(state_dim, hidden_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.num_steps = num_steps
        self.device = device

    def take_action(self, state):
        if isinstance(self.env.action_space, (spaces.Discrete, spaces.MultiDiscrete)):
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            probs = self.actor(state)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            return action.cpu().detach().numpy()
        else:
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            mu, sigma = self.actor(state)
            action_dis = torch.distributions.normal.Normal(mu, sigma)
            action = action_dis.sample()
            return action.cpu().detach().numpy()
        
    def update(self, transition_dict): 
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).to(self.device)
        if isinstance(self.env.action_space, (spaces.Discrete, spaces.MultiDiscrete)):
            actions = actions.unsqueeze(dim=-1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).unsqueeze(dim=-1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        termination = torch.tensor(transition_dict['termination'], dtype=torch.float).unsqueeze(dim=-1).to(self.device)
        trucation = torch.tensor(transition_dict['trucation'], dtype=torch.float).unsqueeze(dim=-1).to(self.device)
        dones = termination 

        # 计算优势函数
        advantage = utils.compute_reward_advantage(self.gamma, self.lmbda, rewards).to(self.device)
        if isinstance(self.env.action_space, (spaces.Discrete, spaces.MultiDiscrete)):
            old_log_probs = torch.log(self.actor(states).gather(-1, actions)).detach()
        else:
            mu, sigma = self.actor(states)
            action_dis = torch.distributions.normal.Normal(mu, sigma)
            old_log_probs = action_dis.log_prob(actions).detach()
        
        for _ in range(self.epochs):
            # 重要性采样比率
            if isinstance(self.env.action_space, (spaces.Discrete, spaces.MultiDiscrete)):
                log_probs = torch.log(self.actor(states).gather(-1, actions))
            else:
                mu, sigma = self.actor(states)
                action_dis = torch.distributions.normal.Normal(mu, sigma)
                log_probs = action_dis.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断

            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
    
    def save_model(self, path, env_name):
        torch.save(self.actor.state_dict(), '{}/actor_{}.pth'.format(path, env_name))
        torch.save(self.critic.state_dict(), '{}/critic_{}.pth'.format(path, env_name))

    def load_model(self, path, env_name):
        self.actor.load_state_dict(torch.load('{}/actor_{}.pth'.format(path, env_name)))
        self.critic.load_state_dict(torch.load('{}/critic_{}.pth'.format(path, env_name)))
        self.actor.eval()
        self.critic.eval()
