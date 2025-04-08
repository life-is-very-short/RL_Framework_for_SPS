import torch
import numpy as np
import torch.nn as nn
import collections
import random  
import matplotlib.pyplot as plt

from matplotlib import animation
from tqdm import tqdm

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

def compute_advantage(gamma, lmbda, td_delta, done):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = np.zeros_like(td_delta[0])
    for i, delta in enumerate(td_delta[::-1]):
        advantage = gamma * lmbda * (1 - done[::-1][i]) * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)

def compute_reward_advantage(rewards):
    r_mean = torch.mean(rewards, dim=0)
    r_std = torch.std(rewards, dim=0) + 0.00001
    rewards = (rewards - r_mean) / r_std
    rewards = rewards.cpu().detach().numpy()
    advantage_list = []
    advantage = np.zeros_like(rewards[0])
    for i, delta in enumerate(rewards[::-1]):
        advantage = advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)

def compute_kl_divergence(ref_log_probs, log_probs):
    gamma = log_probs - ref_log_probs
    return torch.exp(gamma)-gamma-1

def compute_potential_reward(reward, state, next_state):
    pass

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'real_rewards': [], 'termination': [], 'trucation': []}
                state, _ = env.reset(seed = i_episode*23)
                termination = False
                trucation = False
                n_step = 0
                while n_step < agent.num_steps:
                    action = agent.take_action(state)
                    next_state, reward, termination, trucation, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['real_rewards'].append(reward)
                    transition_dict['termination'].append(termination)
                    transition_dict['trucation'].append(trucation)
                    state = next_state
                    n_step += 1
                    episode_return += reward

                return_list.append(episode_return.mean())
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_potential_agent(env, agent, potential_agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 
                                   'rewards': [], 'termination': [], 'trucation': [],
                                   'real_rewards': []}
                state, _ = env.reset(seed = i_episode)
                termination = False
                trucation = False
                n_step = 0
                while n_step < agent.num_steps:
                    action = agent.take_action(state)
                    next_state, real_reward, termination, trucation, _ = env.step(action)
                    reward = potential_agent.reward_shaping(state, next_state, real_reward)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['real_rewards'].append(real_reward)
                    transition_dict['termination'].append(termination)
                    transition_dict['trucation'].append(trucation)
                    state = next_state
                    n_step += 1
                    episode_return += real_reward

                return_list.append(episode_return.mean())
                agent.update(transition_dict)
                potential_agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def display_frames_as_gif(frames, save_path, algo, reward_mode):
    patch = plt.imshow(frames[0][0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i][0])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval = 5)
    anim.save("assets/{}_{}_{}.gif".format(save_path, algo, reward_mode), writer="pillow", fps = 30)

