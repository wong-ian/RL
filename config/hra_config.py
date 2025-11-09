from config.base import BaseConfig

class HRAConfig(BaseConfig):
    # Training loop parameters
    NUM_EPISODES = 20000
    MAX_STEPS = 5000     # Max steps per episode

    # Logging and saving
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 500
    
    # Paths for HRA model and logs
    MODEL_DIR = "models/hra_agent"
    LOG_DIR = "results/hra_agent"

    # HRA-specific hyperparameters from the paper/repo
    REPLAY_MAX_SIZE = 100000  # Size of the experience replay buffer
    REPLAY_MIN_SIZE = 10000   # Min experiences before learning starts
    MINIBATCH_SIZE = 32
    UPDATE_FREQ = 100         # How often to update the target network
    LEARNING_FREQUENCY = 4    # Perform a learning step every N env steps
    
    LEARNING_RATE = 0.001     # Corresponds to 'hra+1' mode
    GAMMA = 0.99              # Discount factor for future rewards
    
    # Epsilon-greedy exploration strategy
    EPSILON_START = 1.0
    EPSILON_FINAL = 0.05
    EPSILON_DECAY_STEPS = 1000000 # Over how many steps to anneal epsilon

    # Number of heads must match the wrapper
    NUM_HEADS = 5