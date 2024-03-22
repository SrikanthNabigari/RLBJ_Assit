import torch
import numpy as np
from tqdm import tqdm
from dqn_agent import DQNAgent
from data_preprocessing import load_data, preprocess_data


# Action mapping
action_mapping = {
    'S': 0,  # Stand
    'H': 1,  # Hit
    'D': 2,  # Double
    'P': 3,  # Split
    'R': 4,  # Surrender
    'I': 5,  # Insurance
    'N': 6   # No Insurance
}

# Load the trained model
model_path = 'trained_models/dqn_agent_epoch_100.pth'  # Replace 'X' with the desired epoch number
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

state_dim = 4  # dealer_up, initial_hand_total, dealer_total, player_total
action_dim = len(action_mapping)  # Number of unique actions

agent = DQNAgent(state_dim, action_dim).to(device)
agent.load_state_dict(torch.load(model_path))
agent.eval()

# Load and preprocess the evaluation dataset
file_path = '/home/srikanth/RL_BlackJack/data/blackjack_simulator.csv'
num_hands_to_evaluate =250000  # Number of hands to use for evaluation
data = load_data(file_path, num_hands=num_hands_to_evaluate)
processed_data = preprocess_data(data)

print("Data loaded and preprocessed.")

# Evaluate the agent's performance
num_episodes = 1000  # Number of episodes to evaluate
num_iterations_per_episode = 1000
total_rewards = []
win_count = 0

for episode in tqdm(range(num_episodes), desc="Evaluating"):
    episode_reward = 0
    print(f"Starting episode {episode + 1}")
    
    for i, row in processed_data.sample(n=num_iterations_per_episode).iterrows():
        dealer_up = np.float32(row['dealer_up'])
        initial_hand_total = np.float32(row['initial_hand_total'])
        dealer_total = np.float32(row['dealer_total'])
        player_total = np.float32(row['player_total'])
        
        state = np.array([dealer_up, initial_hand_total, dealer_total, player_total], dtype=np.float32)
        
        q_values = agent(torch.from_numpy(state).unsqueeze(0).to(device)).detach().cpu().numpy()
        action = np.argmax(q_values)
        
        actions_taken = row['actions_taken']
        if len(actions_taken) > 0:
            last_action = actions_taken[-1]
            reward = row['win'] if action == action_mapping[last_action] else -row['win']
        else:
            reward = 0  # Assign a default reward
        
        episode_reward += reward
        
    
    print(f"Episode {episode + 1} reward: {episode_reward}")
    total_rewards.append(episode_reward)
    if episode_reward > 0:
        win_count += 1

# Calculate evaluation metrics
avg_reward = np.mean(total_rewards)
win_rate = win_count / num_episodes

print(f"Average Reward: {avg_reward:.2f}")
print(f"Win Rate: {win_rate:.2f}")