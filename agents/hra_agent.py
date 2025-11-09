import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque
from agents.base import BaseAgent

# --- Experience Replay Buffer ---
# Stores transitions and allows for efficient sampling of minibatches.
class ExperienceReplayBuffer:
    def __init__(self, max_size, minibatch_size, num_heads):
        self.buffer = deque(maxlen=max_size)
        self.minibatch_size = minibatch_size
        self.num_heads = num_heads

    def add(self, state, action, reward_vec, next_state, done):
        self.buffer.append((state, action, reward_vec, next_state, done))

    def sample(self):
        minibatch = random.sample(self.buffer, self.minibatch_size)
        # Transpose the minibatch
        states, actions, reward_vecs, next_states, dones = map(np.array, zip(*minibatch))
        return states, actions, reward_vecs, next_states, dones

    def __len__(self):
        return len(self.buffer)

# --- HRA Neural Network ---
# A multi-headed architecture with a shared convolutional base.
class HRANetwork(nn.Module):
    def __init__(self, input_shape, num_actions, num_heads):
        super(HRANetwork, self).__init__()
        self.num_heads = num_heads
        
        # Shared base for processing image input (standard DQN CNN architecture)
        self.shared_base = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate the flattened feature size after the conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            feature_size = self.shared_base(dummy_input).shape[1]

        # Create a list of separate output layers ("heads")
        self.heads = nn.ModuleList([
            nn.Linear(feature_size, num_actions) for _ in range(self.num_heads)
        ])

    def forward(self, x):
        # Convert to float and scale pixel values
        x = x.float() / 255.0
        features = self.shared_base(x)
        # Return a list of Q-value tensors, one from each head
        return [head(features) for head in self.heads]

# --- HRA Agent ---
class HRAAgent(BaseAgent):
    def __init__(self, action_space, config):
        super().__init__(action_space, config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"HRAAgent is using device: {self.device}")

        # The observation space for ALE MsPacman is (210, 160, 3). We'll use a wrapper for preprocessing
        # to simplify, e.g., to grayscale (1, 84, 84). Here we assume a preprocessed shape.
        # Note: You MUST apply preprocessing (e.g., via gym wrappers) for this to work.
        # A common preprocessing is AtariPreprocessing from gymnasium.
        self.obs_shape = (1, 84, 84) # Example shape after preprocessing

        self.network = HRANetwork(self.obs_shape, self.action_space.n, self.config.NUM_HEADS).to(self.device)
        self.target_network = HRANetwork(self.obs_shape, self.action_space.n, self.config.NUM_HEADS).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.LEARNING_RATE)
        self.replay_buffer = ExperienceReplayBuffer(
            self.config.REPLAY_MAX_SIZE, self.config.MINIBATCH_SIZE, self.config.NUM_HEADS
        )
        self.total_steps = 0
        self.epsilon = self.config.EPSILON_START

    def get_action(self, observation):
        self.total_steps += 1
        # Update epsilon based on the total number of steps taken
        self.epsilon = max(
            self.config.EPSILON_FINAL,
            self.config.EPSILON_START - (self.total_steps / self.config.EPSILON_DECAY_STEPS)
        )

        if random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
                q_values_list = self.network(obs_tensor)
                # Sum the Q-values from all heads for action selection
                summed_q_values = torch.stack(q_values_list).sum(dim=0)
                return torch.argmax(summed_q_values).item()

    def update(self, state, action, reward, next_state, done, info):
        # The 'reward' parameter from main.py is the total score, which we ignore.
        # We use the decomposed reward vector from our wrapper.
        decomposed_reward = info.get('decomposed_reward', np.zeros(self.config.NUM_HEADS))
        self.replay_buffer.add(state, action, decomposed_reward, next_state, done)

        if len(self.replay_buffer) < self.config.REPLAY_MIN_SIZE:
            return
        
        if self.total_steps % self.config.LEARNING_FREQUENCY == 0:
            self._learn()

        if self.total_steps % self.config.UPDATE_FREQ == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def _learn(self):
        states, actions, rewards_vecs, next_states, dones = self.replay_buffer.sample()

        states = torch.tensor(states, dtype=torch.float32, device=self.device).unsqueeze(1)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_vecs = torch.tensor(rewards_vecs, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values_list = self.network(states)
        
        with torch.no_grad():
            next_q_values_list = self.target_network(next_states)

        total_loss = 0
        for i in range(self.config.NUM_HEADS):
            q_of_action = q_values_list[i].gather(1, actions)
            
            # The original paper uses the mean of next Q-values for the target, which is like Expected SARSA.
            # This is a key detail for faithful reproduction.
            next_q_mean = next_q_values_list[i].mean(dim=1, keepdim=True)

            reward_for_head = rewards_vecs[:, i].unsqueeze(1)
            
            target = reward_for_head + self.config.GAMMA * next_q_mean * (1 - dones.unsqueeze(1))
            
            loss = nn.functional.smooth_l1_loss(q_of_action, target) # Huber loss
            total_loss += loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.network.state_dict(), filepath)
        print(f"HRA model saved to {filepath}")

    def load(self, filepath):
        if not os.path.exists(filepath):
            print(f"Warning: No model found at {filepath}, starting with random weights.")
            return
        self.network.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_network.load_state_dict(self.network.state_dict())
        print(f"HRA model loaded from {filepath}")