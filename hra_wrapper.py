import gymnasium as gym
import numpy as np

# This list defines the "heads" of our agent. The order is critical and must be consistent.
REWARD_HEADS = ['pellet', 'power_pellet', 'eat_ghost', 'fruit', 'death']
REWARD_HEAD_INDICES = {name: i for i, name in enumerate(REWARD_HEADS)}

class HRARewardWrapper(gym.Wrapper):
    """
    This wrapper decomposes the single reward signal from the Ms. Pac-Man environment
    into a vector of rewards, one for each component that HRA will learn about.
    """
    def __init__(self, env):
        super().__init__(env)
        self.last_info = None
        # The decomposed reward vector will be part of the observation space if needed,
        # but we will pass it through the `info` dictionary for clarity.
        print("HRA Reward Wrapper Initialized.")

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        # Initialize last_info with a full state
        self.last_info = {
            'lives': info.get('lives', 0),
            'score': 0  # Assuming score starts at 0
        }
        return observation, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        decomposed_reward = np.zeros(len(REWARD_HEADS), dtype=np.float32)
        score_diff = info.get('score', 0) - self.last_info.get('score', 0)

        # 1. Handle death penalty
        if info.get('lives', 0) < self.last_info.get('lives', 0):
            decomposed_reward[REWARD_HEAD_INDICES['death']] = -100.0  # Assign a large penalty for dying

        # 2. Decompose score difference using game heuristics
        elif score_diff == 10:
            decomposed_reward[REWARD_HEAD_INDICES['pellet']] = 10.0
        elif score_diff == 50:
            decomposed_reward[REWARD_HEAD_INDICES['power_pellet']] = 50.0
        elif score_diff in [200, 400, 800, 1600]:
            decomposed_reward[REWARD_HEAD_INDICES['eat_ghost']] = float(score_diff)
        elif score_diff in [100, 300, 500, 700, 1000, 2000, 3000, 5000]:
            decomposed_reward[REWARD_HEAD_INDICES['fruit']] = float(score_diff)
        
        # Pass the vector through the info dictionary
        info['decomposed_reward'] = decomposed_reward
        
        # Update last_info for the next step
        self.last_info = {
            'lives': info.get('lives', 0),
            'score': info.get('score', 0)
        }

        return obs, reward, terminated, truncated, info