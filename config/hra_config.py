from config.base import BaseConfig

class HRAConfig(BaseConfig):
    # Training loop parameters - reduced for faster training
    NUM_EPISODES = 5000      # Reduced from 20000 for faster initial training
    MAX_STEPS = 5000         # Max steps per episode

    # Logging and saving - unchanged as requested
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 500
    
    # Paths for HRA model and logs
    MODEL_DIR = "models/hra_agent"
    LOG_DIR = "results/hra_agent"

    # HRA-specific hyperparameters optimized for speed and accuracy
    REPLAY_MAX_SIZE = 50000   # Reduced from 100000 for faster startup
    REPLAY_MIN_SIZE = 5000    # Reduced from 10000 for faster learning start
    MINIBATCH_SIZE = 64       # Increased from 32 for better GPU utilization
    UPDATE_FREQ = 100         # How often to update the target network
    LEARNING_FREQUENCY = 2    # Reduced from 4 for more frequent learning
    
    LEARNING_RATE = 0.003     # Increased from 0.001 for faster convergence
    GAMMA = 0.99              # Discount factor for future rewards
    
    # Epsilon-greedy exploration strategy
    EPSILON_START = 1.0
    EPSILON_FINAL = 0.05
    EPSILON_DECAY_STEPS = 1000000 # Over how many steps to anneal epsilon

    # Number of heads must match the wrapper
    NUM_HEADS = 5
