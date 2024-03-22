import torch
import numpy as np
from tqdm import tqdm
from dqn_agent import DQNAgent, DQNTrainer
from experience_replay import ExperienceReplayBuffer
from data_preprocessing import load_data, preprocess_data, split_data
import os

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

if not os.path.exists('trained_models'):
    os.makedirs('trained_models')

# Load and preprocess the dataset
file_path = '/home/srikanth/RL_BlackJack/data/blackjack_simulator.csv'
num_hands_to_load = 5000000
print("Loading dataset...")
data = load_data(file_path, num_hands=num_hands_to_load)
print(f"Loaded {len(data)} hands from the dataset.")

print("Preprocessing data...")
processed_data = preprocess_data(data)
train_data, test_data = split_data(processed_data)
print(f"Preprocessed data. Train set: {len(train_data)} hands, Test set: {len(test_data)} hands.")

# Create the DQN agent, target network, and optimizer
state_dim = 4  # dealer_up, initial_hand_total, dealer_total, player_total
action_dim = len(action_mapping)  # Number of unique actions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

agent = DQNAgent(state_dim, action_dim).to(device)
target_agent = DQNAgent(state_dim, action_dim).to(device)
target_agent.load_state_dict(agent.state_dict())

optimizer = torch.optim.Adam(agent.parameters(), lr=0.0005)

# Create the experience replay buffer
replay_buffer = ExperienceReplayBuffer(capacity=10000)

# Create the DQN trainer
trainer = DQNTrainer(agent, target_agent, optimizer, device)

# Training loop
num_epochs = 200
batch_size = 64
update_freq = 100

print("Starting training...")
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    for _, row in tqdm(train_data.iterrows(), total=len(train_data), desc="Training"):
        dealer_up = np.float32(row['dealer_up'])
        initial_hand_total = np.float32(row['initial_hand_total'])
        dealer_total = np.float32(row['dealer_total'])
        player_total = np.float32(row['player_total'])
        
        state = np.array([dealer_up, initial_hand_total, dealer_total, player_total], dtype=np.float32)
        
        actions_taken = row['actions_taken']
        if len(actions_taken) > 0:
            action_str = actions_taken[-1]  # Get the last action from the 'actions_taken' list
            action = action_mapping[action_str]  # Map the action string to an integer
        else:
            # Handle the case when 'actions_taken' is empty
            action = 0  # Assign a default action (e.g., stand)
        
        reward = row['win']
        next_state = np.array([dealer_up, initial_hand_total, dealer_total, player_total], dtype=np.float32)
        done = 1  # Set done to 1 for each sample since the game is over

        replay_buffer.add(state, action, reward, next_state, done)

    if (epoch + 1) % update_freq == 0:
        trainer.update_target_agent()
        print("Updated target network.")

    # Evaluate the agent on the test dataset
    test_rewards = []
    for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Testing"):
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
            # Handle the case when 'actions_taken' is empty
            reward = 0  # Assign a default reward
        
        test_rewards.append(reward)

    avg_test_reward = np.mean(test_rewards)
    print(f"Test Reward: {avg_test_reward:.2f}")

    # Save the model checkpoint
    torch.save(agent.state_dict(), f"trained_models/dqn_agent_epoch_{epoch + 1}.pth")
    print(f"Saved model checkpoint for epoch {epoch + 1}.")

print("Training completed.")