"""
Train Microsoft's HRA Architecture on Ms. Pac-Man
Updated to save the final model correctly.
"""

import gymnasium as gym
import ale_py
import numpy as np
import os

from agents.hra_agent import MicrosoftHRAAgent
from config.hra_config import HRAConfig
from hra_wrapper import HRARewardWrapper

# Register environments to avoid Namespace errors
gym.register_envs(ale_py)

def main():
    print("="*80)
    print("MICROSOFT HRA TRAINING - DEEP IMPLEMENTATION")
    print("="*80)
    
    # Setup Environment
    env = gym.make('ALE/MsPacman-v5', render_mode=None)
    env = HRARewardWrapper(env)
    
    config = HRAConfig()
    
    # Initialize Agent
    agent = MicrosoftHRAAgent(num_actions=9, config=config)
    
    # Ensure model directory exists
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # Use config number or override here
    total_episodes = 5 
    
    for episode in range(total_episodes):
        obs, info = env.reset()
        agent.total_steps = 0 
        
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(obs, info)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.update(obs, action, reward, next_obs, done, info)
            
            obs = next_obs
            episode_reward += reward
            
        print(f"Episode {episode+1}: Score {episode_reward:.0f} | Steps: {agent.total_steps}")
        
        # Periodic save (optional for short runs)
        if (episode + 1) % 50 == 0:
            save_path = f"{config.MODEL_DIR}/agent_ep{episode+1}.pth"
            agent.save(save_path)
            print(f"Saved checkpoint: {save_path}")

    # --- CRITICAL FIX: SAVE FINAL MODEL ---
    final_save_path = f"{config.MODEL_DIR}/agent_final.pth"
    agent.save(final_save_path)
    print(f"TRAINING FINISHED. Final model saved to: {final_save_path}")
    print("="*80)

    env.close()

if __name__ == "__main__":
    main()