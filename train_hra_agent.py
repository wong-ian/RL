import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing

from main import train
from agents.hra_agent import HRAAgent
from config.hra_config import HRAConfig
from hra_wrapper import HRARewardWrapper
import logging

# Set up basic logging to see INFO messages
logging.basicConfig(level=logging.INFO)

def main():
    logging.info("Starting HRA Agent Training.")

    # 1. Create the base environment
    env = gym.make("ALE/MsPacman-v5")
    
    # 2. Apply standard Atari preprocessing
    env = AtariPreprocessing(env, frame_skip=1, screen_size=84, grayscale_obs=True, scale_obs=False)
    
    # 3. Wrap it with your HRA reward decomposer
    hra_env = HRARewardWrapper(env)
    
    # 4. Pass the agent class, config object, and wrapped env to the train function
    trained_model_path = train(HRAAgent, HRAConfig(), env=hra_env)
    
    logging.info(f"Training complete. Model saved at: {trained_model_path}")
    hra_env.close()

if __name__ == '__main__':
    main()