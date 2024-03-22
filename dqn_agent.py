import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQNAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

class DQNTrainer:
    def __init__(self, agent, target_agent, optimizer, device):
        self.agent = agent
        self.target_agent = target_agent
        self.optimizer = optimizer
        self.device = device

    def train_step(self, states, actions, rewards, next_states, dones, gamma=0.99):
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)

        q_values = self.agent(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_agent(next_states).max(1)[0]
        target_q_values = rewards + gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_agent(self):
        self.target_agent.load_state_dict(self.agent.state_dict())