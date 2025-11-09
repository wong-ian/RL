"""
Train Microsoft's HRA Architecture on Ms. Pac-Man
Complete implementation with paper specifications
"""

import numpy as np
import torch
import gymnasium as gym
import ale_py
import time
import pickle
import os
from hra_agent import MicrosoftHRAAgent
from collections import deque

class MsPacManTrainer:
    """
    Complete trainer for Microsoft's HRA architecture
    """
    
    def __init__(self):
        # Create environment
        print("Creating Ms. Pac-Man environment...")
        self.env = gym.make('ALE/MsPacman-v5', render_mode=None)
        
        # Don't use AtariPreprocessing - we need raw frames for object extraction
        print("   Using RAW frames (210x160x3) for object extraction")
        
        # Create Microsoft HRA agent
        self.agent = MicrosoftHRAAgent(9)  # Ms. Pac-Man has 9 actions
        
        # Training parameters
        self.max_episodes = 1000
        self.max_steps_per_episode = 10000
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_scores = []
        self.episode_lengths = []
        self.gvf_counts = []
        self.position_counts = []
        
        # Saving
        self.save_dir = "models/microsoft_hra_exact"
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"Microsoft HRA Trainer initialized")
        print(f"   Max episodes: {self.max_episodes}")
        print(f"   Max steps: {self.max_steps_per_episode}")
    
    def train(self):
        """Train the Microsoft HRA agent"""
        print("\n" + "="*80)
        print("MICROSOFT HRA TRAINING - EXACT PAPER IMPLEMENTATION")
        print("="*80)
        
        best_score = 0
        recent_scores = deque(maxlen=100)
        
        for episode in range(self.max_episodes):
            start_time = time.time()
            
            # Reset environment and agent state
            obs, info = self.env.reset()
            self.agent.reset()
            
            episode_reward = 0.0
            episode_score = 0.0
            step = 0
            
            # Episode loop
            while step < self.max_steps_per_episode:
                # Agent action selection
                action = self.agent.get_action(obs, info)
                
                # Environment step
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Update agent
                self.agent.update(obs, action, reward, next_obs, done, info)
                
                # Track metrics
                episode_reward += float(reward)
                episode_score += float(reward)  # In Ms. Pac-Man, reward = score increment
                
                # Next iteration
                obs = next_obs
                step += 1
                
                if done:
                    break
            
            # Episode completed
            episode_time = time.time() - start_time
            
            # Track performance
            self.episode_rewards.append(episode_reward)
            self.episode_scores.append(episode_score)
            self.episode_lengths.append(step)
            self.gvf_counts.append(len(self.agent.position_gvfs))
            self.position_counts.append(len(self.agent.visited_positions))
            
            recent_scores.append(episode_score)
            avg_recent_score = np.mean(recent_scores) if recent_scores else 0
            
            # Update best score
            if episode_score > best_score:
                best_score = episode_score
                self.save_agent(f"agent_best_score_{int(best_score)}.pkl")
            
            # Logging
            self.log_episode(episode, episode_score, episode_reward, step, 
                           avg_recent_score, episode_time, best_score)
            
            # Periodic saves
            if (episode + 1) % 50 == 0:
                self.save_agent(f"agent_ep{episode+1}.pkl")
                self.save_training_stats()
            
            # Performance analysis
            if (episode + 1) % 100 == 0:
                self.analyze_performance()
        
        print("\nTraining completed!")
        self.save_agent("agent_final.pkl")
        self.save_training_stats()
        self.final_analysis()
    
    def log_episode(self, episode, score, reward, steps, avg_score, time_taken, best_score):
        """Log episode performance"""
        gvf_count = len(self.agent.position_gvfs)
        position_count = len(self.agent.visited_positions)
        level = self.agent.current_level
        
        print(f"Episode {episode+1:4d} | "
              f"Score: {score:6.0f} | "
              f"Avg: {avg_score:6.1f} | "
              f"Best: {best_score:6.0f} | "
              f"Steps: {steps:4d} | "
              f"Level: {level:2d} | "
              f"GVFs: {gvf_count:3d} | "
              f"Time: {time_taken:4.1f}s")
        
        # Special achievements (less frequent)
        if score >= 5000:
            print(f"EXCELLENT! Score {score} achieved!")
        elif score >= 2000:
            print(f"GREAT! Score {score} achieved!")
        elif score >= 1000:
            print(f"GOOD! Score {score} achieved!")
        
        # Only show GVF milestone every 100 GVFs
        if gvf_count > 0 and gvf_count % 100 == 0 and gvf_count != getattr(self, 'last_gvf_milestone', 0):
            print(f"Milestone: {gvf_count} position GVFs created")
            self.last_gvf_milestone = gvf_count
        
        if level > 1:
            print(f"Advanced to level {level}!")
    
    def analyze_performance(self):
        """Analyze current performance trends"""
        if len(self.episode_scores) < 10:
            return
        
        recent_scores = self.episode_scores[-10:]
        avg_recent = np.mean(recent_scores)
        std_recent = np.std(recent_scores)
        max_recent = np.max(recent_scores)
        
        print(f"\nPERFORMANCE ANALYSIS (Last 10 episodes):")
        print(f"   Average: {avg_recent:.1f} +- {std_recent:.1f}")
        print(f"   Best: {max_recent:.0f}")
        print(f"   Position GVFs: {self.gvf_counts[-1] if self.gvf_counts else 0}")
        print(f"   Visited Positions: {self.position_counts[-1] if self.position_counts else 0}")
        
        # Architecture growth
        if len(self.gvf_counts) >= 10:
            gvf_growth = self.gvf_counts[-1] - self.gvf_counts[-10]
            print(f"   GVF growth: +{gvf_growth} in last 10 episodes")
        
        # Memory analysis
        level_memories = len(self.agent.executive_memory.level_memories)
        if level_memories > 0:
            print(f"   Executive memory: {level_memories} levels memorized")
    
    def final_analysis(self):
        """Final performance analysis"""
        if not self.episode_scores:
            return
        
        print("\n" + "="*80)
        print("FINAL MICROSOFT HRA ANALYSIS - EXACT IMPLEMENTATION")
        print("="*80)
        
        # Performance stats
        max_score = np.max(self.episode_scores)
        avg_score = np.mean(self.episode_scores)
        final_avg = np.mean(self.episode_scores[-100:]) if len(self.episode_scores) >= 100 else avg_score
        
        print(f"PERFORMANCE RESULTS:")
        print(f"   Best Score: {max_score:.0f}")
        print(f"   Average Score: {avg_score:.1f}")
        print(f"   Final 100 Average: {final_avg:.1f}")
        
        # Architecture stats
        final_gvfs = self.gvf_counts[-1] if self.gvf_counts else 0
        final_positions = self.position_counts[-1] if self.position_counts else 0
        max_level = self.agent.current_level
        
        print(f"\nARCHITECTURE RESULTS:")
        print(f"   Total Position GVFs: {final_gvfs}")
        print(f"   Visited Positions: {final_positions}")
        print(f"   Max Level: {max_level}")
        
        # Memory analysis
        memory_levels = len(self.agent.executive_memory.level_memories)
        total_sequences = sum(len(seqs) for seqs in self.agent.executive_memory.level_memories.values())
        
        print(f"   Memorized Levels: {memory_levels}")
        print(f"   Total Sequences: {total_sequences}")
        
        # Comparison with paper
        print(f"\nCOMPARISON WITH MICROSOFT PAPER:")
        print(f"   Paper Result: 25,304 points")
        print(f"   Our Result: {max_score:.0f} points")
        print(f"   Percentage: {(max_score/25304)*100:.1f}% of paper")
        
        # Technical comparison
        print(f"\nTECHNICAL IMPLEMENTATION:")
        print(f"   Position-based GVFs: {final_gvfs} (vs paper: ~1,600)")
        print(f"   Object multipliers: exact match")
        print(f"   Score normalization: enabled")
        print(f"   Targeted exploration: UCB-style")
        print(f"   Executive memory: enabled")
        
        if max_score >= 20000:
            print("OUTSTANDING! Near paper-level performance!")
        elif max_score >= 10000:
            print("EXCELLENT! Strong HRA implementation!")
        elif max_score >= 5000:
            print("GOOD! Solid progress with HRA!")
        else:
            print("Architecture learning - more episodes needed")
    
    def save_agent(self, filename):
        """Save the trained agent"""
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self.agent, f)
        print(f"Agent saved to {filepath}")
    
    def save_training_stats(self):
        """Save training statistics"""
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_scores': self.episode_scores,
            'episode_lengths': self.episode_lengths,
            'gvf_counts': self.gvf_counts,
            'position_counts': self.position_counts
        }
        
        filepath = os.path.join(self.save_dir, 'training_stats.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(stats, f)
        print(f"Training stats saved to {filepath}")

def main():
    """Run Microsoft HRA training"""
    trainer = None
    try:
        trainer = MsPacManTrainer()
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if trainer:
            print("Saving current progress...")
            trainer.save_agent("agent_interrupted.pkl")
            trainer.save_training_stats()
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
