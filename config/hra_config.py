from config.base import BaseConfig

class HRAConfig(BaseConfig):
    # Microsoft HRA training parameters
    NUM_EPISODES = 5000       
    MAX_STEPS = 10000         
    
    MODEL_DIR = "models/hra_agent"
    LOG_DIR = "results/hra_agent"

    # Hyperparameters
    REPLAY_MAX_SIZE = 10000
    MINIBATCH_SIZE = 32
    
    LEARNING_RATE = 0.0001     # Lower LR for stability
    GAMMA = 0.99
    
    # HRA Architecture
    NUM_HEADS = 5 
    USE_NORMALIZATION = True   # CRITICAL