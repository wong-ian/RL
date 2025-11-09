from abc import ABC

class BaseConfig(ABC):    
    # Training parameters
    NUM_EPISODES: int
    MAX_STEPS: int
    
    # Logging
    LOG_INTERVAL: int
    SAVE_INTERVAL: int
    
    # Paths
    MODEL_DIR: str
    LOG_DIR: str