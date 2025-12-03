"""
Microsoft HRA - EXACT Deep Implementation
Refactored for Spatial Training & Orientation Tracking
With MPS (Mac) Support and Pink Pellet Fix
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import random
from collections import deque
from hra_network import HRAMsPacmanNetwork

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, orientation, action, next_state, next_orientation, done):
        self.buffer.append((state, orientation, action, next_state, next_orientation, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, orient, action, next_state, next_orient, done = zip(*batch)
        return (np.array(state), np.array(orient), np.array(action), 
                np.array(next_state), np.array(next_orient), np.array(done))
    
    def __len__(self):
        return len(self.buffer)

class MsPacManObjectExtractor:
    def __init__(self):
        # Using the colors calibrated for your environment
        self.colors = {
            'pacman': [210, 164, 74],      # Standard Gold
            'ghost_red': [200, 72, 72],
            'ghost_pink': [180, 122, 48],
            'ghost_cyan': [84, 184, 153],
            'ghost_orange': [198, 108, 58],
            'ghost_blue': [66, 114, 194],
            'fruit': [184, 70, 162],
            
            # --- FIXED COLOR: PINKISH PELLET ---
            'pellet': [228, 111, 111], 
            # -----------------------------------
        }
    
    def extract_objects(self, frame):
        # Crop to 160x160 (remove top 25 and bottom 25)
        cropped = frame[25:185, :]
        
        # Resize to 40x40
        resized = cv2.resize(cropped, (40, 40), interpolation=cv2.INTER_NEAREST)
        
        # Initialize channels (11, 40, 40)
        channels = np.zeros((11, 40, 40), dtype=np.float32)
        
        # Channel 0: Ms. Pac-Man
        channels[0] = self._detect_color(resized, self.colors['pacman'])
        
        # Channels 1-4: Individual ghosts
        channels[1] = self._detect_color(resized, self.colors['ghost_red'])
        channels[2] = self._detect_color(resized, self.colors['ghost_pink'])
        channels[3] = self._detect_color(resized, self.colors['ghost_cyan'])
        channels[4] = self._detect_color(resized, self.colors['ghost_orange'])
        
        # Channel 5: Blue ghosts (edible)
        channels[5] = self._detect_color(resized, self.colors['ghost_blue'])
        
        # Channel 9: Fruit
        channels[9] = self._detect_color(resized, self.colors['fruit'])
        
        # Channel 10: Pellets (Using the new Pink color)
        # We use a higher tolerance (50) to catch slightly darker/lighter pinks
        channels[10] = self._detect_color(resized, self.colors['pellet'], tolerance=50)
        
        return channels

    def _detect_color(self, frame, target_color, tolerance=40):
        diff = np.abs(frame.astype(np.int32) - np.array(target_color))
        mask = np.all(diff < tolerance, axis=2)
        return mask.astype(np.float32)

class MicrosoftHRAAgent:
    def __init__(self, num_actions, config=None):
        self.num_actions = num_actions
        
        # --- MPS SUPPORT ADDED HERE ---
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("HRA Agent running on: CUDA (NVIDIA)")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("HRA Agent running on: MPS (Mac Metal Acceleration)")
        else:
            self.device = torch.device("cpu")
            print("HRA Agent running on: CPU (Warning: Slow)")
        # ------------------------------

        self.lr = 0.0001
        self.gamma = 0.99
        self.use_normalization = True 
        
        self.object_extractor = MsPacManObjectExtractor()
        self.model = HRAMsPacmanNetwork(num_actions=num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(capacity=20000)
        self.batch_size = 32
        
        # Orientation Tracking (N, E, S, W)
        self.last_orientation = np.array([0, 1, 0, 0], dtype=np.float32) 
        self.last_pos = (20, 20)
        
        self.total_steps = 0
        self.pacman_pos_idx = (20, 20)
        self.visit_counts = np.zeros((40, 40, 9), dtype=np.int32)

    def check_vision(self, env):
        """Debug method to ensure agent sees objects"""
        print("Checking Vision System...")
        obs, _ = env.reset()
        channels = self.object_extractor.extract_objects(obs)
        
        p_count = np.sum(channels[10])
        pac_count = np.sum(channels[0])
        g_count = np.sum(channels[1:5])
        
        print(f"  Detected Pellets Pixels: {p_count}")
        print(f"  Detected Pacman Pixels: {pac_count}")
        print(f"  Detected Ghost Pixels: {g_count}")
        
        if p_count == 0:
            print("  WARNING: Agent is BLIND to Pellets! Check color tolerance.")
        if pac_count == 0:
            print("  WARNING: Agent is BLIND to Self! Check color tolerance.")
        
    def preprocess(self, obs):
        return self.object_extractor.extract_objects(obs)

    def _update_orientation(self, new_pos):
        y, x = new_pos
        old_y, old_x = self.last_pos
        
        dy = y - old_y
        dx = x - old_x
        
        if abs(dy) > 0 or abs(dx) > 0:
            new_orient = np.zeros(4, dtype=np.float32)
            if abs(dy) > abs(dx): # Moved vertical
                if dy < 0: new_orient[0] = 1.0 # North
                else: new_orient[2] = 1.0      # South
            else: # Moved horizontal
                if dx > 0: new_orient[1] = 1.0 # East
                else: new_orient[3] = 1.0      # West
            self.last_orientation = new_orient
            self.last_pos = new_pos

    def get_aggregated_q_values(self, state_channels, orientation, pacman_pos_idx):
        state_t = torch.FloatTensor(state_channels).unsqueeze(0).to(self.device)
        orient_t = torch.FloatTensor(orientation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            gvf_vals, ghost_vals, blue_vals, fruit_vals = self.model(state_t, orient_t)
            
        # 1. POSITIVE Rewards (Spatial Aggregation)
        pellet_mask = torch.FloatTensor(state_channels[10]).to(self.device)
        
        # Q_pellet = Sum(GVF_map * Pellet_mask)
        pellet_q = (gvf_vals * pellet_mask.unsqueeze(0).unsqueeze(0)).sum(dim=(2, 3)) * 10.0
        
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

        # 3. NEGATIVE Rewards (Ghosts)
        ghost_weight = -2.0 
        total_negative_q = ghost_vals.squeeze(0).sum(dim=1) * ghost_weight
        
        # 4. EXPLORATION
        div_q = torch.zeros(self.num_actions).to(self.device)
        if self.total_steps < 50: 
            div_q = torch.rand(self.num_actions).to(self.device) * 5.0 
            
        y, x = pacman_pos_idx
        y = min(max(y, 0), 39)
        x = min(max(x, 0), 39)
        counts = self.visit_counts[y, x, :]
        
        exploration_bonus = np.sqrt(self.total_steps / (counts + 1.0)) * 0.05
        exp_q = torch.FloatTensor(exploration_bonus).to(self.device)
        
        final_q = norm_positive_q + total_negative_q + div_q + exp_q
        return final_q

    def get_action(self, obs, info=None):
        channels = self.preprocess(obs)
        
        # Find Pacman
        indices = np.where(channels[0] > 0.5)
        if len(indices[0]) > 0:
            self.pacman_pos_idx = (indices[0][0], indices[1][0])
            self._update_orientation(self.pacman_pos_idx)
        
        final_q = self.get_aggregated_q_values(channels, self.last_orientation, self.pacman_pos_idx)
        action = torch.argmax(final_q).item()
        
        y, x = self.pacman_pos_idx
        self.visit_counts[y, x, action] += 1
        return action

    def update(self, obs, action, reward, next_obs, done, info):
        self.total_steps += 1
        
        state = self.preprocess(obs)
        next_state = self.preprocess(next_obs)
        
        self.memory.push(state, self.last_orientation, action, next_state, self.last_orientation, done)
        
        if len(self.memory) < 1000: return
            
        # --- SPATIAL TRAINING STEP ---
        states, orients, actions, next_states, next_orients, dones = self.memory.sample(self.batch_size)
        
        states_t = torch.FloatTensor(states).to(self.device)
        orients_t = torch.FloatTensor(orients).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        next_orients_t = torch.FloatTensor(next_orients).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # 1. Forward Pass
        gvf, ghosts, blue, fruit = self.model(states_t, orients_t)
        
        with torch.no_grad():
            next_gvf, next_ghost, next_blue, next_fruit = self.model(next_states_t, next_orients_t)

        total_loss = 0
        mse = nn.MSELoss()

        # --- A. PELLET GVF (Spatial) ---
        reward_map_pellet = next_states_t[:, 10, :, :] # (B, 40, 40)
        next_spatial_val = next_gvf.max(dim=1)[0] # (B, 40, 40)
        
        spatial_target = reward_map_pellet + (self.gamma * next_spatial_val * (1 - dones_t.unsqueeze(1).unsqueeze(1)))
        
        curr_spatial_val = gvf.gather(1, actions_t.view(-1, 1, 1, 1).expand(-1, 1, 40, 40)).squeeze(1)
        
        loss_pellet = mse(curr_spatial_val, spatial_target)
        total_loss += loss_pellet

        # --- B. GHOST HEAD (Spatial Avoidance) ---
        ghost_map = next_states_t[:, 1:5, :, :].sum(dim=1) # (B, 40, 40)
        ghost_present = ghost_map.sum(dim=(1,2)) > 0 # (B)
        ghost_target_scalar = torch.where(ghost_present, torch.tensor(-1.0).to(self.device), torch.tensor(0.0).to(self.device))
        
        curr_ghost_val = ghosts.sum(dim=2).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        next_ghost_val = next_ghost.sum(dim=2).min(dim=1)[0]
        target_ghost = ghost_target_scalar + (self.gamma * next_ghost_val * (1 - dones_t))
        
        loss_ghost = mse(curr_ghost_val, target_ghost)
        total_loss += loss_ghost

        # Optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
