from config.base import BaseConfig

class RandomConfig(BaseConfig):    
    NUM_EPISODES = 10
    MAX_STEPS = 100
    
    # Logging
    LOG_INTERVAL = 1
    SAVE_INTERVAL = 1
    
    # Paths
    MODEL_DIR = "models/random"
    LOG_DIR = "results/random"