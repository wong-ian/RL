# Use this to test your models.

import os
from main import train, play
from agents.random_agent import RandomAgent
from config.random_config import RandomConfig
from logging import getLogger

# Disable the SDL audio driver warnings so there's no console warning output fo rit
os.environ['SDL_AUDIODRIVER'] = 'dummy'
logger = getLogger(__name__)

def main():
    logger.info("In main.")
    _ = train(RandomAgent, RandomConfig, False)
    play(RandomAgent, RandomConfig, '', num_episodes=1)

if __name__ == '__main__':
    main()