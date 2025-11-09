# Agent Plays Ms. Pac-Man

ECS 170 Group 9

## Why Ms. Pac-Man?
I think we read somewhere that it was better for an AI agent? Ms. Pac-Man also has more variety in terms of the actual game itself. There are four unique mazes, as opposed to Pac-Man's one single maze, and the ghosts in Ms. Pac-Man are more randomized and unpredictable, so it would be more challenging for us to design a model. This led us to settle on Ms. Pac-Man over the original Pac-Man.

## Environment
We're using the Atari Learning Environment (ALE) and Gymnasium to function as our environment for the agent. We're developing 

## Prerequisites

- Python 3.8 or later
- pip (usually included with Python)

## Create and activate a virtual environment (Linux / macOS)

1. Create the virtual environment:

```
python3 -m venv .venv
```

2. Activate it:

```
source .venv/bin/activate
```

On Windows (PowerShell):

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Install dependencies with pip

After activating the virtual environment, upgrade pip and install the project's dependencies:

```
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Deactivate the virtual environment

When you're done working:

```
deactivate
```

## If `requirements.txt` is missing

You can create it from an active environment:

```
pip freeze > requirements.txt
```

---

These steps set up an isolated Python environment and install all dependencies using pip.

## Project Structure

```
pacman/
├── agents/
│   ├── base.py             # Abstract base class
│   └── my_agent.py         # Your custom agent
├── config/
│   ├── base.py             # Base config class
│   └── my_config.py        # Your agent's config
├── models/
│   └── my_agent/           # Saved checkpoints (.pkl files)
├── results/
│   └── my_agent/           # Training logs
├── main.py                 # Training & playing functions
├── train_my_agent.py       # Your training script
└── play_my_agent.py        # Your visualization script
└── test.py                 # Example of testing script (for a random agent that doesn't learn)
```

## Functions in main.py

### `train(agent_class, config, render_human=False)`
- **agent_class**: Your custom agent class (not an instance)
- **config**: Your custom configuration object
- **render_human**: Set to `True` to watch training visually (wayyyy slower, do not recommend)
- **Returns**: Path to saved model

### `play(agent_class, config, model_path, num_episodes=5)`
- **agent_class**: Your custom agent class
- **config**: Your custom configuration object
- **model_path**: Path to saved .pkl file (you can leave it as an empty string for untrained model). This was generated inside `train()`
- **num_episodes**: Number of games to play