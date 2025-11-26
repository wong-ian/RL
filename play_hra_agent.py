"""
Play Microsoft HRA Deep Agent
Updated for Directory Structure
"""
import gymnasium as gym
import ale_py
import torch
import time
import numpy as np
import os

# --- CORRECTED IMPORTS ---
from agents.hra_agent import MicrosoftHRAAgent
from config.hra_config import HRAConfig
# -------------------------
gym.register_envs(ale_py)

def main():
    # 1. Configuration
    config = HRAConfig()
    render_mode = "human" # Set to None for faster non-visual speed
    
    # 2. Setup Environment
    env = gym.make('ALE/MsPacman-v5', render_mode=render_mode)
    
    # 3. Load Agent
    agent = MicrosoftHRAAgent(num_actions=9, config=config)
    
    # Find the latest model
    model_dir = config.MODEL_DIR
    if os.path.exists(model_dir):
        models = sorted([f for f in os.listdir(model_dir) if f.endswith('.pth')])
        if models:
            # Sort by episode number if possible, otherwise by name
            # This lambda extracts numbers from 'agent_ep50.pth' -> 50
            models.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
            
            latest_model = os.path.join(model_dir, models[-1])
            print(f"Loading model: {latest_model}")
            agent.load(latest_model)
        else:
            print("No models found in directory. Playing with random weights.")
    else:
        print(f"Model directory '{model_dir}' not found. Playing with random weights.")

    # 4. Play Loop
    num_episodes = 3
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            if render_mode == "human":
                time.sleep(0.02)
        
        print(f"Episode {episode+1} Score: {total_reward}")

    env.close()

if __name__ == "__main__":
    main()