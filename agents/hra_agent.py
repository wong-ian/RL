"""
Microsoft HRA - EXACT Deep Implementation
Refactored based on Research Report
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import random
from collections import deque

# Assumes hra_network.py is in the root directory
from hra_network import HRAMsPacmanNetwork

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, orientation, action, rewards, next_state, next_orientation, done):
        self.buffer.append((state, orientation, action, rewards, next_state, next_orientation, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, orient, action, reward, next_state, next_orient, done = zip(*batch)
        return (np.array(state), np.array(orient), np.array(action), 
                np.array(reward), np.array(next_state), np.array(next_orient), np.array(done))
    
    def __len__(self):
        return len(self.buffer)

class MsPacManObjectExtractor:
    def __init__(self):
        self.colors = {
            'pacman': [210, 164, 74],
            'ghost_red': [200, 72, 72],
            'ghost_pink': [180, 122, 48],
            'ghost_cyan': [84, 184, 153],
            'ghost_orange': [198, 108, 58],
            'ghost_blue': [66, 114, 194],
            'pellet': [228, 111, 111],
            'fruit': [184, 70, 162],
        }
    
    def extract_objects(self, frame):
        cropped = frame[25:185, :]
        resized = cv2.resize(cropped, (40, 40), interpolation=cv2.INTER_NEAREST)
        channels = np.zeros((11, 40, 40), dtype=np.float32)
        
        channels[0] = self._detect_color(resized, self.colors['pacman'])
        channels[1] = self._detect_color(resized, self.colors['ghost_red'])
        channels[2] = self._detect_color(resized, self.colors['ghost_pink'])
        channels[3] = self._detect_color(resized, self.colors['ghost_cyan'])
        channels[4] = self._detect_color(resized, self.colors['ghost_orange'])
        channels[5] = self._detect_color(resized, self.colors['ghost_blue'])
        channels[9] = self._detect_color(resized, self.colors['fruit'])
        channels[10] = self._detect_pellets(resized)
        
        return channels

    def _detect_color(self, frame, target_color, tolerance=30):
        diff = np.abs(frame.astype(np.int32) - np.array(target_color))
        mask = np.all(diff < tolerance, axis=2)
        return mask.astype(np.float32)
    
    def _detect_pellets(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        bright_mask = gray > 200
        return bright_mask.astype(np.float32)

class MicrosoftHRAAgent:
    def __init__(self, num_actions, config=None):
        self.num_actions = num_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"HRA Agent running on: {self.device}")

        self.lr = config.LEARNING_RATE if config else 0.0001
        self.gamma = config.GAMMA if config else 0.99
        self.use_normalization = True 
        
        self.object_extractor = MsPacManObjectExtractor()
        self.model = HRAMsPacmanNetwork(num_actions=num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(capacity=10000)
        self.batch_size = 32
        
        self.last_orientation = np.array([1, 0, 0, 0], dtype=np.float32)
        self.total_steps = 0
        self.pacman_pos_idx = (20, 20)
        self.visit_counts = np.zeros((40, 40, 9), dtype=np.int32)
        
    def preprocess(self, obs):
        return self.object_extractor.extract_objects(obs)

    def get_pacman_pos(self, channels):
        indices = np.where(channels[0] > 0.5)
        if len(indices[0]) > 0:
            return (indices[0][0], indices[1][0])
        return self.pacman_pos_idx

    def get_aggregated_q_values(self, state_channels, orientation, pacman_pos_idx):
        """
        Calculates aggregated Q-values using Normalization 
        and Exploration Heads.
        """
        state_t = torch.FloatTensor(state_channels).unsqueeze(0).to(self.device)
        orient_t = torch.FloatTensor(orientation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            gvf_vals, ghost_vals, blue_vals, fruit_vals = self.model(state_t, orient_t)
            
        # 1. POSITIVE Rewards
        pellet_mask = torch.FloatTensor(state_channels[10]).to(self.device)
        # GVF (Navigation) * Pellet Mask = Value of eating pellets
        pellet_q = (gvf_vals * pellet_mask.unsqueeze(0).unsqueeze(0) * 10.0).sum(dim=(2, 3)) 
        
        fruit_exists = state_channels[9].max() > 0
        fruit_q = fruit_vals.squeeze(0).squeeze(1) if fruit_exists else torch.zeros(self.num_actions).to(self.device)
        blue_ghost_q = blue_vals.squeeze(0).sum(dim=1)
        
        total_positive_q = pellet_q.squeeze(0) + fruit_q + blue_ghost_q
        
        # 2. NORMALIZATION
        if self.use_normalization:
            min_q = total_positive_q.min()
            max_q = total_positive_q.max()
            span = max_q - min_q
            if span > 1e-6:
                norm_positive_q = (total_positive_q - min_q) / span
            else:
                norm_positive_q = torch.zeros_like(total_positive_q)
        else:
            norm_positive_q = total_positive_q

        # 3. NEGATIVE Rewards
        ghost_weight = -10.0 if self.use_normalization else -1000.0
        total_negative_q = ghost_vals.squeeze(0).sum(dim=1) * ghost_weight
        
        # 4. EXPLORATION
        div_q = torch.zeros(self.num_actions).to(self.device)
        if self.total_steps < 50:
            div_q = torch.rand(self.num_actions).to(self.device) * 20.0
            
        y, x = pacman_pos_idx
        y = min(max(y, 0), 39)
        x = min(max(x, 0), 39)
        counts = self.visit_counts[y, x, :]
        
        exploration_bonus = np.sqrt(self.total_steps / (counts + 1.0)) * 0.1
        exp_q = torch.FloatTensor(exploration_bonus).to(self.device)
        
        final_q = norm_positive_q + total_negative_q + div_q + exp_q
        return final_q

    def get_action(self, obs, info=None):
        channels = self.preprocess(obs)
        self.pacman_pos_idx = self.get_pacman_pos(channels)
        orientation = self.last_orientation
        final_q = self.get_aggregated_q_values(channels, orientation, self.pacman_pos_idx)
        action = torch.argmax(final_q).item()
        
        y, x = self.pacman_pos_idx
        self.visit_counts[y, x, action] += 1
        return action

    def update(self, obs, action, reward_decomposed, next_obs, done, info):
        self.total_steps += 1
        
        state = self.preprocess(obs)
        next_state = self.preprocess(next_obs)
        orientation = self.last_orientation
        next_orientation = self.last_orientation
        
        if 'decomposed_reward' in info:
             # Wrapper returns: ['pellet', 'power_pellet', 'eat_ghost', 'fruit', 'death']
             rewards = info['decomposed_reward'] 
        else:
             rewards = np.zeros(5)
             rewards[0] = float(reward_decomposed)
             
        self.memory.push(state, orientation, action, rewards, next_state, next_orientation, done)
        
        if len(self.memory) < 1000:
            return
            
        # --- Real Training Step ---
        states, orients, actions, rewards, next_states, next_orients, dones = self.memory.sample(self.batch_size)
        
        states_t = torch.FloatTensor(states).to(self.device)
        orients_t = torch.FloatTensor(orients).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)     # (Batch)
        rewards_t = torch.FloatTensor(rewards).to(self.device)    # (Batch, 5)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        next_orients_t = torch.FloatTensor(next_orients).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)        # (Batch)

        # 1. Get Current Q-Values for all heads
        # Shapes: GVF(B,9,40,40), Ghost(B,9,4), Blue(B,9,4), Fruit(B,9,1)
        gvf, ghosts, blue, fruit = self.model(states_t, orients_t)

        # 2. Get Next State Q-Values (for bootstrapping)
        with torch.no_grad():
            next_gvf, next_ghost, next_blue, next_fruit = self.model(next_states_t, next_orients_t)

        # --- LOSS CALCULATION PER HEAD ---
        total_loss = 0
        
        # A. PELLET HEAD (GVF)
        # Reward Index 0 is Pellet
        curr_q_pellet = gvf.mean(dim=(2,3)).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        next_q_pellet = next_gvf.mean(dim=(2,3)).max(1)[0]
        target_pellet = rewards_t[:, 0] + (self.gamma * next_q_pellet * (1 - dones_t))
        loss_pellet = nn.MSELoss()(curr_q_pellet, target_pellet)
        total_loss += loss_pellet

        # B. GHOST HEAD (Avoidance)
        # Reward Index 4 is Death (-100). 
        # Ghost head outputs 4 values (one per ghost). We sum them for a "Total Threat" estimate.
        curr_q_ghost = ghosts.sum(dim=2).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        next_q_ghost = next_ghost.sum(dim=2).max(1)[0]
        # Use Death penalty (index 4)
        target_ghost = rewards_t[:, 4] + (self.gamma * next_q_ghost * (1 - dones_t))
        loss_ghost = nn.MSELoss()(curr_q_ghost, target_ghost)
        total_loss += loss_ghost

        # C. FRUIT HEAD
        # Reward Index 3 is Fruit
        curr_q_fruit = fruit.squeeze(2).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        next_q_fruit = next_fruit.squeeze(2).max(1)[0]
        target_fruit = rewards_t[:, 3] + (self.gamma * next_q_fruit * (1 - dones_t))
        loss_fruit = nn.MSELoss()(curr_q_fruit, target_fruit)
        total_loss += loss_fruit

        # Optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path))