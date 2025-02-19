import gym
import random
import math
import numpy as np
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define a simple transition tuple for experience replay.
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        """Save a transition."""
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Define the dynamic data-driven neuron controller module.
class DynamicNeuronController(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DynamicNeuronController, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        # Compute an intermediate representation.
        x_mod = F.relu(self.fc1(x))
        # Generate modulation factors in the range [-1, 1].
        modulation = torch.tanh(self.fc2(x_mod))
        return modulation

# Define the Q-network that integrates the dynamic neuron controller.
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DQN, self).__init__()
        self.controller = DynamicNeuronController(state_dim, hidden_dim)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        # Apply the dynamic controller to get modulation factors.
        modulation = self.controller(x)
        # Modulate the input state element-wise.
        x_modulated = x * modulation
        x_hidden = F.relu(self.fc1(x_modulated))
        q_values = self.fc2(x_hidden)
        return q_values

def select_action(state, policy_net, steps_done, device, n_actions, 
                  EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200):
    """Select an action using the epsilon-greedy strategy."""
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # Return the action with the highest Q-value.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model(policy_net, target_net, memory, optimizer, batch_size, device, gamma=0.99):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
    expected_state_action_values = (next_state_values * gamma) + reward_batch.squeeze()
    
    loss = F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def moving_average(data, window_size):
    """Compute the moving average of the data using the specified window size."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def main():
    env = gym.make('CartPole-v1')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]
    
    policy_net = DQN(state_dim, n_actions).to(device)
    target_net = DQN(state_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    memory = ReplayMemory(10000)
    
    num_episodes = 500
    batch_size = 128
    target_update = 10
    steps_done = 0

    # Lists to store training logs.
    episode_rewards = []
    episode_timesteps = []
    
    for i_episode in range(num_episodes):
        # For the latest Gym API, reset returns (observation, info).
        state_np, _ = env.reset()
        state = torch.tensor([state_np], device=device, dtype=torch.float32)
        total_reward = 0
        timesteps = 0
        
        while True:
            timesteps += 1
            action = select_action(state, policy_net, steps_done, device, n_actions)
            steps_done += 1
            
            # For the latest Gym API, step returns (next_state, reward, done, truncated, info).
            next_state_np, reward, done, truncated, _ = env.step(action.item())
            done = done or truncated
            total_reward += reward
            reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)
            
            if done:
                next_state = None
            else:
                next_state = torch.tensor([next_state_np], device=device, dtype=torch.float32)
            
            memory.push(state, action, next_state, reward_tensor)
            state = next_state if next_state is not None else torch.tensor([next_state_np], device=device, dtype=torch.float32)
            optimize_model(policy_net, target_net, memory, optimizer, batch_size, device)
            
            if done:
                break
        
        print(f"Episode {i_episode} finished after {timesteps} timesteps, total reward: {total_reward}")
        episode_rewards.append(total_reward)
        episode_timesteps.append(timesteps)
        
        if i_episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
    env.close()
    
    # -------------------------
    # Visualization Section
    # -------------------------
    
    episodes = np.arange(len(episode_rewards))
    
    # Plot total reward and timesteps per episode.
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(episodes, episode_rewards, marker='o', linestyle='-', markersize=3, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode vs Total Reward")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(episodes, episode_timesteps, marker='o', linestyle='-', markersize=3, color='orange', label="Timesteps")
    plt.xlabel("Episode")
    plt.ylabel("Timesteps")
    plt.title("Episode vs Timesteps")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot the moving average of the total rewards.
    window_size = 10
    ma_rewards = moving_average(episode_rewards, window_size)
    plt.figure(figsize=(8, 6))
    plt.plot(episodes[:len(ma_rewards)], ma_rewards, color='green', label=f"Moving Average (window={window_size})")
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Reward")
    plt.title("Moving Average of Total Reward")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
