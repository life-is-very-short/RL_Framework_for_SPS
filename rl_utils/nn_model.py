import numpy as np
import torch
import torch.nn as nn

from gymnasium import spaces
from torch.nn import functional as F

class PolicyNet_Discrete(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet_Discrete, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)
    
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc1 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.net(x)
        return self.fc1(x)
    
class PolicyNet_Continue(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet_Continue, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, 1)
        self.sigma = nn.Linear(hidden_dim, 1)


    def forward(self, x):
        x = self.net(x)
        mu = torch.tanh(self.mu(x)) * 2
        sigma = F.softplus(self.sigma(x)) + 0.00001
        return mu, sigma
