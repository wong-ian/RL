# Use this to test your models.
import os
import argparse
from main import train as main_train, play as main_play
from logging import getLogger

from agents.random_agent import RandomAgent
from config.random_config import RandomConfig
from agents.deep_ql_agent import DQNAgent, DQNConfig

# PPO imports
from train_ppo import train_ppo_agent, play_ppo_agent, train as ppo_train
from config.ppo_config import PPOConfig

# Disable the SDL audio driver warnings so there's no console warning output fo rit
os.environ['SDL_AUDIODRIVER'] = 'dummy'
logger = getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train or play a Pac-Man agent.")
    parser.add_argument("--agent", type=str, required=True, help="Specify the agent to use (e.g., 'random', 'dqn', 'ppo')")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "play", "both"], 
                       help="Mode: 'train' only, 'play' only, or 'both' (train then play)")
    parser.add_argument("--model-path", type=str, default="", 
                       help="Path to model file for 'play' mode (required if mode is 'play')")
    parser.add_argument("--episodes", type=int, default=5, 
                       help="Number of episodes to play (for 'play' or 'both' modes)")
    parser.add_argument("--resume-from", type=str, default="", 
                       help="Where to resume training from")
    args = parser.parse_args()

    agent_to_run = None
    config_to_use = None
    train_func = main_train
    play_func = main_play

    if args.agent.lower() == 'random':
        agent_to_run = RandomAgent
        config_to_use = RandomConfig()
    elif args.agent.lower() == 'dqn':
        agent_to_run = DQNAgent
        config_to_use = DQNConfig()
    elif args.agent.lower() == 'ppo':
        agent_to_run = None  # PPO uses its own internal agent
        config_to_use = PPOConfig()
        train_func = train_ppo_agent
        play_func = play_ppo_agent
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    model_path = args.model_path
    resume_from = args.resume_from

    if args.mode == "train" or args.mode == "both":
        logger.info(f"Running training for agent: {args.agent}")
        
        # Handle resume functionality for PPO
        if args.agent.lower() == 'ppo' and resume_from:
            if not os.path.exists(resume_from):
                logger.error(f"Resume checkpoint not found: {resume_from}")
                return
            
            logger.info(f"Resuming PPO training from checkpoint: {resume_from}")
            model_path = ppo_train(cfg=config_to_use, resume_from=resume_from)
        else:
            # Normal training (no resume)
            if resume_from and args.agent.lower() != 'ppo':
                logger.warning(f"Resume functionality only supported for PPO agent. Ignoring --resume-from for {args.agent}")
            model_path = train_func(agent_to_run, config_to_use, render_human=False)
        
        logger.info(f"Training complete. Model saved to: {model_path}")
    
    if args.mode == "play" or args.mode == "both":
        if not model_path:
            logger.error("No model path provided for play mode!")
            logger.error("Use --model-path <path> or run in 'both' mode to train first")
            return
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return
        
        logger.info(f"Starting a play session with {args.episodes} episodes...")
        play_func(agent_to_run, config_to_use, model_path, num_episodes=args.episodes)

if __name__ == '__main__':
    main()