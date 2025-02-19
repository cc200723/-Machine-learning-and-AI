import matplotlib.pyplot as plt
import numpy as np
import os

def parse_training_results(file_path):
    """
    Parses a training log file with lines formatted as:
    "Episode X finished after Y timesteps, total reward: Z"
    and returns three lists: episodes, timesteps, and rewards.
    """
    episodes = []
    timesteps = []
    rewards = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("Episode"):
                parts = line.strip().split()
                try:
                    # Example line:
                    # Episode 0 finished after 15 timesteps, total reward: 15.0
                    episode = int(parts[1])
                    timestep = int(parts[4])
                    # The reward is expected as the last token.
                    reward = float(parts[-1])
                    episodes.append(episode)
                    timesteps.append(timestep)
                    rewards.append(reward)
                except Exception as e:
                    print("Error parsing line:", line, e)
    return episodes, timesteps, rewards

def moving_average(data, window_size):
    """
    Computes the moving average using a simple convolution.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def visualize_results(file_path):
    """
    Loads the training log from file_path, parses the data,
    and generates the following visualizations:
      1. Total reward per episode.
      2. Timesteps per episode.
      3. Moving average of total reward.
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Please provide the training results file.")
        return

    episodes, timesteps, rewards = parse_training_results(file_path)
    
    # Convert lists to numpy arrays for easier manipulation.
    episodes = np.array(episodes)
    timesteps = np.array(timesteps)
    rewards = np.array(rewards)
    
    # Create a figure with two subplots: one for total reward and one for timesteps.
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Total Reward per Episode
    plt.subplot(1, 2, 1)
    plt.plot(episodes, rewards, marker='o', linestyle='-', markersize=3, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode vs Total Reward")
    plt.legend()
    
    # Plot 2: Timesteps per Episode
    plt.subplot(1, 2, 2)
    plt.plot(episodes, timesteps, marker='o', linestyle='-', markersize=3, color='orange', label="Timesteps")
    plt.xlabel("Episode")
    plt.ylabel("Timesteps")
    plt.title("Episode vs Timesteps")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Moving Average of Total Reward
    window_size = 10
    ma_rewards = moving_average(rewards, window_size)
    plt.figure(figsize=(8, 6))
    plt.plot(episodes[:len(ma_rewards)], ma_rewards, color='green', label=f"Moving Average (window={window_size})")
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Reward")
    plt.title("Moving Average of Total Reward")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Specify the path to your training log file.
    file_path = "D://desktop//training_results.txt"
    visualize_results(file_path)
