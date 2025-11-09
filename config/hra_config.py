from config.base import BaseConfig

class HRAConfig(BaseConfig):
    # Microsoft HRA training parameters - FULL TRAINING CONFIGURATION  
    NUM_EPISODES = 5000       # Microsoft's full training: total_eps = 5000
    MAX_STEPS = 10000         # Remove artificial cap - let games run naturally
    
    # Microsoft HRA training structure 
    EPS_PER_EPOCH = 10       # Microsoft uses 10 episodes per epoch
    MAX_START_NULLOPS = 0    # Microsoft uses 0
    
    # Logging and saving 
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 50        # Save every 50 episodes    # Paths for HRA model and logs
    MODEL_DIR = "models/hra_agent"
    LOG_DIR = "results/hra_agent"

    # Microsoft HRA hyperparameters - exact match to config.yaml
    REPLAY_MAX_SIZE = 10000   # replay_max_size
    REPLAY_MIN_SIZE = 1000    # replay_min_size
    MINIBATCH_SIZE = 32       # minibatch_size
    UPDATE_FREQ = 100         # update_freq
    LEARNING_FREQUENCY = 4    # learning_frequency
    HISTORY_LEN = 1           # history_len
    
    LEARNING_RATE = 0.001     # learning_rate
    GAMMA = 0.99              # Microsoft default (not in config but in utils.py)
    
    # Microsoft epsilon strategy - ADAPTED FOR ATARI COMPLEXITY
    EPSILON_START = 1.0       # Start with exploration
    EPSILON_FINAL = 0.1       # Reduce to 10% random actions (vs Microsoft's 100%)
    EPSILON_DECAY_STEPS = 1000  # Decay over first 1000 episodes (vs Microsoft's no decay)
    TEST_EPSILON = 0.05       # Lower test epsilon for better evaluation    # Microsoft HRA network architecture adapted for Atari
    NUM_UNITS = 250              # Microsoft default for dense networks (250 total / 5 heads = 50 per head)
    ACTION_DIM = 1               # action_dim
    
    # Microsoft HRA mode settings
    USE_MEAN = True           # use_mean: True for HRA
    USE_HRA = True            # use_hra: True
    REMOVE_FEATURES = False   # remove_features: False for standard HRA
    
    # Random seed for reproducibility
    RANDOM_SEED = 1234        # random_seed

    # Microsoft HRA architecture adapted for Atari Ms. Pac-Man
    # Number of heads - matches Ms. Pac-Man reward structure (not Fruit Collection)
    NUM_HEADS = 5                # Ms. Pac-Man: pellet, power_pellet, eat_ghost, fruit, death
    
    # Microsoft RMSprop parameters - EXACT MATCH from ai.py
    RMSPROP_ALPHA = 0.95      # Microsoft: rho = 0.95  
    RMSPROP_EPS = 1e-7        # Microsoft: epsilon = 1e-7
