import torch
import torch.nn as nn
import torch.nn.functional as F

class HRAMsPacmanNetwork(nn.Module):
    """
    The Hybrid Reward Architecture Network for Ms. Pac-Man.
    
    This architecture separates the value estimation into:
    1. Navigation GVFs: A vectorized output predicting values for reaching ANY grid location.
    2. Entity Heads: Specific heads for Ghosts, Blue Ghosts, and Fruit.
    
    Inputs:
        x: (Batch, 11, 40, 40) - The 11-channel binary representation.
        orientation: (Batch, 4) - One-hot encoded direction.
    """
    def __init__(self, input_channels=11, num_actions=9, map_height=40, map_width=40):
        super(HRAMsPacmanNetwork, self).__init__()
        self.num_actions = num_actions
        self.map_h = map_height
        self.map_w = map_width
        
        # --- Shared Feature Extractor ---
        # The 'Torso' of the network. 
        # Processes the spatial grid to extract features relevant for all heads.
        
        # Conv1: 11 in -> 16 out, kernel 3x3, stride 1, padding 1
        # Maintains 40x40 spatial resolution.
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        
        # Conv2: 16 in -> 32 out, kernel 3x3, stride 2, padding 1
        # Downsamples to 20x20.
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        
        # Conv3: 32 in -> 32 out, kernel 3x3, stride 2, padding 1
        # Downsamples to 10x10.
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        
        # Flattened size: 32 channels * 10 * 10 = 3200 features.
        self.flattened_size = 32 * 10 * 10
        self.orientation_size = 4
        
        # Shared Dense Layer
        # Fuses visual features with orientation.
        self.fc_shared = nn.Linear(self.flattened_size + self.orientation_size, 512)
        
        # --- HEAD 1: Vectorized GVF (Navigation) ---
        # Instead of 1600 separate heads for each tile, we output a dense map.
        # Output shape: num_actions * map_height * map_width
        # This predicts Q(s,a) for reaching location (h, w).
        self.gvf_layer = nn.Linear(512, num_actions * map_height * map_width)
        
        # --- HEAD 2: Ghosts (Threats) ---
        # 4 distinct ghosts. Each needs a Q-value for each action.
        self.ghost_head = nn.Linear(512, num_actions * 4) 
        
        # --- HEAD 3: Blue Ghosts (Opportunities) ---
        # 4 distinct blue ghosts.
        self.blue_ghost_head = nn.Linear(512, num_actions * 4)
        
        # --- HEAD 4: Fruit (Bonus) ---
        # 1 fruit type active at a time usually.
        self.fruit_head = nn.Linear(512, num_actions)

    def forward(self, x, orientation):
        """
        Forward pass generating decomposed Q-values.
        """
        batch_size = x.size(0)
        
        # 1. Feature Extraction
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        
        h = h.view(batch_size, -1) # Flatten
        
        # 2. Orientation Fusion
        # Ensure orientation matches batch size
        if orientation.size(0) != batch_size:
             orientation = orientation.expand(batch_size, -1)
             
        h = torch.cat([h, orientation], dim=1)
        
        # 3. Shared Latent Representation
        embedding = F.relu(self.fc_shared(h))
        
        # 4. Generate Decomposed Values
        
        # GVF Map: Reshape to (Batch, Actions, Height, Width)
        gvf_flat = self.gvf_layer(embedding)
        gvf_values = gvf_flat.view(batch_size, self.num_actions, self.map_h, self.map_w)
        
        # Specific Entity Heads
        ghost_values = self.ghost_head(embedding).view(batch_size, self.num_actions, 4)
        blue_ghost_values = self.blue_ghost_head(embedding).view(batch_size, self.num_actions, 4)
        fruit_values = self.fruit_head(embedding).view(batch_size, self.num_actions, 1)
        
        return gvf_values, ghost_values, blue_ghost_values, fruit_values