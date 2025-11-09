import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing

from main import play
from agents.hra_agent import HRAAgent
from config.hra_config import HRAConfig
from hra_wrapper import HRARewardWrapper
import logging

logging.basicConfig(level=logging.INFO)

def main():
    logging.info("Visualizing HRA Agent.")

    # The model path should point to a saved checkpoint, e.g., the final one
    model_path = "models/hra_agent/agent_ep500.pkl" 

    env = gym.make("ALE/MsPacman-v5", render_mode="human")
    env = AtariPreprocessing(env, frame_skip=1, screen_size=84, grayscale_obs=True, scale_obs=False)
    hra_env = HRARewardWrapper(env) # Good practice to use the same wrappers for consistency

    play(HRAAgent, HRAConfig(), model_path, num_episodes=5, env=hra_env)

    hra_env.close()

if __name__ == '__main__':
    main()