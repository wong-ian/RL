"""
Play Microsoft HRA Clean Agent
Load and visualize the trained Microsoft HRA agent
"""

import gymnasium as gym
import ale_py
import pickle
import time
import numpy as np


def play_microsoft_hra(model_path, num_episodes=5, render=True):
    """Play episodes with trained Microsoft HRA agent"""
    
    print(f"Loading Microsoft HRA agent from: {model_path}")
    
    try:
        # Load the trained agent
        with open(model_path, 'rb') as f:
            agent = pickle.load(f)
        
        print("Agent loaded successfully.")
        print(f"  Position GVFs: {len(agent.position_gvfs)}")
        print(f"  Executive memory levels: {len(agent.executive_memory.level_memories)}")
        print(f"  Current level: {agent.current_level}")
        
    except FileNotFoundError:
        print(f"Error: Model file not found: {model_path}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create environment (with or without rendering)
    render_mode = "human" if render else None
    env = gym.make('ALE/MsPacman-v5', render_mode=render_mode)
    
    print(f"\nPlaying {num_episodes} episodes...")
    
    total_scores = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_score = 0.0
        steps = 0
        max_steps = 20000
        
        # Reset agent's episode state (but keep learned knowledge)
        agent.episode_step = 0
        agent.level_step = 0
        
        print(f"\nEpisode {episode + 1}:")
        episode_start = time.time()
        
        while steps < max_steps:
            # Agent selects action
            action = agent.get_action(obs, info)
            
            # Environment step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update agent (learning continues)
            agent.update(obs, action, reward, next_obs, done, info)
            
            episode_score += float(reward)
            steps += 1
            obs = next_obs
            
            # Add small delay for human viewing
            if render:
                time.sleep(0.02)
            
            if done:
                break
        
        episode_time = time.time() - episode_start
        total_scores.append(episode_score)
        
        print(f"   Score: {episode_score:.0f}")
        print(f"   Steps: {steps}")
        print(f"   Level: {agent.current_level}")
        print(f"   Time: {episode_time:.1f}s")
    
    env.close()
    
    # Summary statistics
    print("\nPERFORMANCE SUMMARY:")
    print(f"  Best episode: {max(total_scores):.0f}")
    print(f"  Average: {np.mean(total_scores):.1f}")
    print(f"  Worst episode: {min(total_scores):.0f}")
    print(f"  Final GVFs: {len(agent.position_gvfs)}")


def list_available_models():
    """List all available Microsoft HRA models"""
    import os
    
    model_dir = "models/microsoft_hra_exact"
    
    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found: {model_dir}")
        return []
    
    model_files = []
    for file in os.listdir(model_dir):
        if file.endswith('.pkl') and 'agent_' in file:
            model_files.append(file)
    
    if model_files:
        print(f"Available models in {model_dir}:")
        for i, model in enumerate(sorted(model_files), 1):
            file_path = os.path.join(model_dir, model)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"  {i}. {model} ({file_size:.1f} MB)")
    else:
        print(f"Error: No model files found in {model_dir}")
    
    return model_files


def main():
    """Main function with model selection"""
    print("Microsoft HRA Agent Player")
    print("=" * 50)
    
    # List available models
    models = list_available_models()
    
    if not models:
        print("\nTrain a model first with: python train_microsoft_hra_exact.py")
        return
    
    # Try to find best model automatically
    best_models = [m for m in models if 'agent_best_' in m]
    if best_models:
        # Sort by score (extract number from filename)
        def extract_score(filename):
            try:
                return int(filename.split('_')[-1].split('.')[0])
            except:
                return 0
        
        best_models.sort(key=extract_score, reverse=True)
        selected_model = best_models[0]
        model_path = f"models/microsoft_hra_exact/{selected_model}"
        
        print(f"\nAuto-selected best model: {selected_model}")
        
    else:
        # Fallback to any available model
        selected_model = models[0]
        model_path = f"models/microsoft_hra_exact/{selected_model}"
        print(f"\nUsing available model: {selected_model}")
    
    # Play episodes
    try:
        play_microsoft_hra(model_path, num_episodes=3, render=True)
    except KeyboardInterrupt:
        print("\nPlayback interrupted.")
    except Exception as e:
        print(f"\nPlayback error: {e}")


if __name__ == "__main__":
    main()
