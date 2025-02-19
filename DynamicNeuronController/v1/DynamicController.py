import gym
import random
import math
import numpy as np
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define a simple transition tuple for experience replay
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
# This module computes modulation factors based on the input state.
class DynamicNeuronController(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DynamicNeuronController, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        # Compute an intermediate representation
        x_mod = F.relu(self.fc1(x))
        # Generate modulation factors in the range [-1, 1]
        modulation = torch.tanh(self.fc2(x_mod))
        return modulation

# Define the Q-network that incorporates the dynamic neuron controller.
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DQN, self).__init__()
        # Dynamic neuron controller module
        self.controller = DynamicNeuronController(state_dim, hidden_dim)
        # Main network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        # Compute modulation from the controller and apply it element-wise
        modulation = self.controller(x)
        x_modulated = x * modulation
        # Pass the modulated state through the network
        x_hidden = F.relu(self.fc1(x_modulated))
        q_values = self.fc2(x_hidden)
        return q_values

# Epsilon-greedy policy for action selection
def select_action(state, policy_net, steps_done, device, n_actions, 
                  EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # Choose the action with highest Q-value
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # Choose a random action
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

# Optimize the model using a batch of transitions from the replay memory.
def optimize_model(policy_net, target_net, memory, optimizer, batch_size, device, gamma=0.99):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    
    # Create a mask for non-final states and concatenate batches
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    # Compute Q(s_t, a) for the actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # Compute the expected Q values for the next states using the target network
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
    expected_state_action_values = (next_state_values * gamma) + reward_batch.squeeze()
    
    # Compute the Huber loss
    loss = F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    # Create the CartPole environment
    env = gym.make('CartPole-v1')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]
    
    # Initialize the policy and target networks
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
    
    for i_episode in range(num_episodes):
        # Updated reset: unpack observation and ignore info
        state_np, _ = env.reset()
        state = torch.tensor([state_np], device=device, dtype=torch.float32)
        total_reward = 0
        for t in range(1, 10000):  # Limit maximum steps per episode
            action = select_action(state, policy_net, steps_done, device, n_actions)
            steps_done += 1
            
            # Updated step: unpack five values and combine done flags
            next_state_np, reward, done, truncated, _ = env.step(action.item())
            done = done or truncated
            total_reward += reward
            reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)
            
            if done:
                next_state = None
            else:
                next_state = torch.tensor([next_state_np], device=device, dtype=torch.float32)
            
            # Store the transition in memory
            memory.push(state, action, next_state, reward_tensor)
            
            # Move to the next state
            state = next_state if next_state is not None else torch.tensor([next_state_np], device=device, dtype=torch.float32)
            
            # Perform one step of optimization on the model
            optimize_model(policy_net, target_net, memory, optimizer, batch_size, device)
            
            if done:
                print(f"Episode {i_episode} finished after {t} timesteps, total reward: {total_reward}")
                break
        
        # Update the target network every few episodes
        if i_episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
    print("Training completed.")
    env.close()

if __name__ == '__main__':
    main()
