"""
Train Microsoft's HRA Architecture on Ms. Pac-Man
With CSV Logging and Vision Sanity Check
"""

import gymnasium as gym
import ale_py
import numpy as np
import os
import csv
import time

from agents.hra_agent import MicrosoftHRAAgent
from config.hra_config import HRAConfig
from hra_wrapper import HRARewardWrapper

# Register environments
gym.register_envs(ale_py)

def main():
    print("="*80)
    print("MICROSOFT HRA TRAINING - REVISED SPATIAL")
    print("="*80)
    
    # Setup
    env = gym.make('ALE/MsPacman-v5', render_mode=None)
    env = HRARewardWrapper(env)
    config = HRAConfig()
    
    # Initialize Agent
    agent = MicrosoftHRAAgent(num_actions=9, config=config)
    
    # --- 1. SANITY CHECK ---
    agent.check_vision(env)
    # -----------------------

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Setup CSV Logger
    csv_path = os.path.join(config.LOG_DIR, 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Episode', 'Score', 'Steps', 'Avg_Q_Pos', 'Avg_Q_Neg'])
    
    total_episodes = 500 # Set to desired amount
    
    for episode in range(total_episodes):
        obs, info = env.reset()
        agent.total_steps = 0 
        
        episode_reward = 0
        done = False
        
        # Tracking metrics for CSV
        q_pos_sum = 0
        q_neg_sum = 0
        steps_taken = 0
        
        while not done:
            # Check for dead pacman to reset orientation if needed? 
            # HRA handles this naturally via state
            
            action = agent.get_action(obs, info)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.update(obs, action, reward, next_obs, done, info)
            
            # Debug Stats
            # We can't easily get Q-values out of update, so we approximate or skip
            # Could return them from get_action if needed.
            
            obs = next_obs
            episode_reward += reward
            steps_taken += 1
            
        print(f"Episode {episode+1}: Score {episode_reward:.0f} | Steps: {steps_taken}")
        
        # Log to CSV
        csv_writer.writerow([episode+1, episode_reward, steps_taken, 0, 0])
        csv_file.flush() # Ensure data is written
        
        if (episode + 1) % 50 == 0:
            save_path = f"{config.MODEL_DIR}/agent_ep{episode+1}.pth"
            agent.save(save_path)
            print(f"Saved checkpoint: {save_path}")

    final_save_path = f"{config.MODEL_DIR}/agent_final.pth"
    agent.save(final_save_path)
    print(f"TRAINING FINISHED. Log saved to {csv_path}")
    
    csv_file.close()
    env.close()

if __name__ == "__main__":
    main()
