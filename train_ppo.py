"""
Training script for PPO with CNN architecture.
Includes Atari preprocessing: frame stacking, grayscale, resizing.
Uses vectorized environments for efficient training.
"""

import os
import math
import time
import random
import logging
import csv
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

import gymnasium as gym
from gymnasium.wrappers import (
    AtariPreprocessing,
    TransformReward,
    FrameStackObservation,
)
from gymnasium.vector import AsyncVectorEnv, AutoresetMode

import ale_py
import matplotlib.pyplot as plt

from config.ppo_config import PPOConfig

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RawScoreWrapper(gym.Wrapper):
    """Wrapper to track raw game scores before reward clipping/scaling."""
    def __init__(self, env):
        super().__init__(env)
        self.raw_episode_score = 0.0
        self.episode_score = 0.0  # Accumulates across lives
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.raw_episode_score += reward
        self.episode_score += reward
        
        # Always store current accumulated score in info
        info['raw_score'] = self.episode_score
        
        # Reset only on actual game over (truncated usually means time limit or actual game end)
        if terminated or truncated:
            # Check if this is actual game over vs just life loss
            if truncated or (terminated and info.get('lives', 0) == 0):
                # True game over - keep the score in info for logging
                info['episode_score'] = self.episode_score
                self.episode_score = 0.0
                self.raw_episode_score = 0.0
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self.raw_episode_score = 0.0
        self.episode_score = 0.0
        return self.env.reset(**kwargs)


def make_single_env(cfg: PPOConfig, render: bool = False):
    def _thunk():
        env = gym.make(
            cfg.ENV_ID,
            frameskip=1,
            repeat_action_probability=cfg.STICKY_PROB,
            render_mode="human" if render else None,
        )

        # Wrap to track raw scores before reward transformation
        env = RawScoreWrapper(env)

        env = AtariPreprocessing(
            env,
            screen_size=84,
            grayscale_obs=True,
            frame_skip=cfg.FRAME_SKIP,
            terminal_on_life_loss=True,
        )

        env = FrameStackObservation(env, stack_size=cfg.FRAME_STACK, padding_type="reset")
        
        # CHANGED: Replaced np.sign (clipping) with scaling (0.01).
        # MsPacman: Dot=10 (0.1), Ghost=200 (2.0). 
        # This preserves the "hunting" incentive.
        env = TransformReward(env, lambda r: 0.01 * r)
        
        return env
    return _thunk


def make_vector_env(cfg: PPOConfig, render_first_env: bool = False) -> AsyncVectorEnv:
    thunks = [make_single_env(cfg, render=(render_first_env and i == 0)) for i in range(cfg.NUM_ENVS)]
    env = AsyncVectorEnv(thunks, autoreset_mode=AutoresetMode.NEXT_STEP)
    return env


class ACNetCNN(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU())
        self.pi = nn.Linear(512, action_dim)
        self.v = nn.Linear(512, 1)

        # Orthogonal initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.constant_(self.pi.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dtype != torch.float32:
            x = x.float()
        x = x / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        logits = self.pi(x)
        value = self.v(x).squeeze(-1)
        return logits, value


class PPOAgent:
    def __init__(self, cfg: PPOConfig, action_dim: int):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg.DEVICE = str(self.device)  # Update config
        self.net = ACNetCNN(action_dim).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.LR, eps=1e-5)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.USE_AMP and self.device.type == "cuda"))

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        logits, value = self.net(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, values = self.net(obs)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        return logp, entropy, values


class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, obs_shape: Tuple[int, ...], device: torch.device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.obs = torch.zeros((num_steps, num_envs) + obs_shape, dtype=torch.uint8, device=device)
        self.actions = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
        self.rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)
        self.logp = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.step_idx = 0

    def add(self, obs, actions, rewards, dones, logp, values):
        i = self.step_idx
        self.obs[i].copy_(obs)
        self.actions[i].copy_(actions)
        self.rewards[i].copy_(rewards)
        self.dones[i].copy_(dones)
        self.logp[i].copy_(logp)
        self.values[i].copy_(values)
        self.step_idx += 1

    def reset(self):
        self.step_idx = 0


def plot_rewards(rewards_history: List[float], log_dir: str, update: int, total_updates: int, is_final: bool = False):
    try:
        if len(rewards_history) < 2:
            return
            
        plt.figure(figsize=(12, 6))
        plt.plot(rewards_history, alpha=0.3, label="Episode reward", color='blue')
        
        # Add moving average
        window_size = 100 if len(rewards_history) >= 100 else 10
        ma = np.convolve(rewards_history, np.ones(window_size) / window_size, mode="valid")
        plt.plot(np.arange(window_size - 1, len(rewards_history)), ma, 
                label=f"Moving average ({window_size})", linewidth=2, color='orange')
        
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Reward", fontsize=12)
        
        if is_final:
            plt.title("Final Training Rewards (PPO-CNN)", fontsize=14)
        else:
            plt.title(f"Training Rewards - Update {update}/{total_updates} ({len(rewards_history)} episodes)", fontsize=14)
        
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        
        # Save plot
        if is_final:
            out_path = os.path.join(log_dir, "rewards_final.png")
            dpi = 150
        else:
            out_path = os.path.join(log_dir, f"rewards_update_{update}.png")
            # Also save as latest for easy viewing
            latest_path = os.path.join(log_dir, "rewards_latest.png")
            dpi = 100
        
        plt.savefig(out_path, bbox_inches="tight", dpi=dpi)
        if not is_final:
            plt.savefig(latest_path, bbox_inches="tight", dpi=dpi)
        
        plt.close()
            
    except Exception as e:
        logger.warning(f"Plotting failed: {e}")


def compute_gae(cfg: PPOConfig, rewards, values, dones, last_values):
    T, B = rewards.shape
    advantages = torch.zeros((T, B), dtype=torch.float32, device=rewards.device)
    gae = torch.zeros((B,), dtype=torch.float32, device=rewards.device)

    for t in reversed(range(T)):
        not_done = (~dones[t]).float()
        next_value = last_values if t == T - 1 else values[t + 1]
        delta = rewards[t] + cfg.GAMMA * next_value * not_done - values[t]
        gae = delta + cfg.GAMMA * cfg.GAE_LAMBDA * not_done * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def train(cfg: PPOConfig = None, resume_from: str = None):
    if cfg is None:
        cfg = PPOConfig()
    
    seed_everything(cfg.SEED)
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

    # Initialize CSV logging
    csv_path = os.path.join(cfg.LOG_DIR, "training_log.csv")
    episode_csv_path = os.path.join(cfg.LOG_DIR, "episode_log.csv")
    
    # Check if resuming to append to existing CSV
    csv_exists = resume_from is not None and os.path.exists(csv_path)
    episode_csv_exists = resume_from is not None and os.path.exists(episode_csv_path)
    
    csv_file = open(csv_path, 'a' if csv_exists else 'w', newline='')
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow([
            'update', 'episodes_completed', 'avg_reward', 'avg_score',
            'policy_loss', 'value_loss', 'entropy', 'clip_frac', 'kl'
        ])
    
    episode_csv_file = open(episode_csv_path, 'a' if episode_csv_exists else 'w', newline='')
    episode_csv_writer = csv.writer(episode_csv_file)
    if not episode_csv_exists:
        episode_csv_writer.writerow([
            'episode', 'update', 'raw_score', 'clipped_reward', 'steps'
        ])

    # Vector env
    env = make_vector_env(cfg, render_first_env=False)
    seeds = [cfg.SEED + i for i in range(cfg.NUM_ENVS)]
    obs_np, _ = env.reset(seed=seeds)
    obs_shape = obs_np.shape[1:]  # (4,84,84)
    action_dim = env.single_action_space.n

    assert len(obs_shape) == 3 and obs_shape[0] == cfg.FRAME_STACK, f"Unexpected obs shape {obs_shape}"

    agent = PPOAgent(cfg, action_dim)
    
    # Resume from checkpoint if specified
    start_update = 1
    rewards_history: List[float] = []
    scores_history: List[float] = []
    episode_count = 0
    
    if resume_from is not None and os.path.exists(resume_from):
        logger.info(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=agent.device)
        
        # Load model weights
        agent.net.load_state_dict(checkpoint["model"])
        logger.info("Loaded model weights")
        
        # Load optimizer state if available
        if "optimizer" in checkpoint:
            agent.opt.load_state_dict(checkpoint["optimizer"])
            logger.info("Loaded optimizer state")
        
        # Load training progress if available
        if "update" in checkpoint:
            start_update = checkpoint["update"] + 1
            logger.info(f"Resuming from update {start_update}")
        
        if "rewards_history" in checkpoint:
            rewards_history = checkpoint["rewards_history"]
            logger.info(f"Loaded {len(rewards_history)} episode rewards from history")
        
        if "scores_history" in checkpoint:
            scores_history = checkpoint["scores_history"]
            logger.info(f"Loaded {len(scores_history)} episode scores from history")
        
        if "episode_count" in checkpoint:
            episode_count = checkpoint["episode_count"]
        
        # Load scaler state if available (for mixed precision)
        if "scaler" in checkpoint and cfg.USE_AMP:
            agent.scaler.load_state_dict(checkpoint["scaler"])
            logger.info("Loaded gradient scaler state")

    # Episode reward tracking
    ep_returns = np.zeros(cfg.NUM_ENVS, dtype=np.float32)
    ep_lengths = np.zeros(cfg.NUM_ENVS, dtype=np.int32)

    # Rollout storage
    buf = RolloutBuffer(cfg.ROLLOUT_STEPS, cfg.NUM_ENVS, obs_shape, agent.device)
    obs = torch.as_tensor(obs_np, device=agent.device)

    for update in range(start_update, cfg.TOTAL_UPDATES + 1):
        # NEW: Linear Learning Rate Decay
        # This helps the agent settle into a good policy in later stages
        frac = 1.0 - (update - 1.0) / cfg.TOTAL_UPDATES
        lrnow = frac * cfg.LR
        agent.opt.param_groups[0]["lr"] = lrnow

        buf.reset()

        # Collect rollout
        for _t in range(cfg.ROLLOUT_STEPS):
            with torch.no_grad():
                actions, logp, values = agent.act(obs)

            actions_np = actions.cpu().numpy()
            next_obs_np, r, terminated, truncated, infos = env.step(actions_np)

            rewards_t = torch.as_tensor(r, dtype=torch.float32, device=agent.device)
            dones_t = torch.as_tensor(terminated, dtype=torch.bool, device=agent.device)

            buf.add(obs, actions, rewards_t, dones_t, logp, values)

            # Episode accounting
            ep_returns += r
            ep_lengths += 1
            done_mask = terminated | truncated
            if np.any(done_mask):
                for i, d in enumerate(done_mask):
                    if d:
                        episode_count += 1
                        clipped_reward = float(ep_returns[i])
                        episode_steps = int(ep_lengths[i])
                        
                        # Get raw score from info
                        raw_score = 0.0
                        if 'final_info' in infos and infos['final_info'][i]:
                            # Try to get episode_score (full game score) or raw_score (current score)
                            if 'episode_score' in infos['final_info'][i]:
                                raw_score = float(infos['final_info'][i]['episode_score'])
                            elif 'raw_score' in infos['final_info'][i] and infos['final_info'][i]['raw_score'] is not None:
                                raw_score = float(infos['final_info'][i]['raw_score'])
                        
                        rewards_history.append(clipped_reward)
                        scores_history.append(raw_score)
                        
                        # Log episode to CSV
                        episode_csv_writer.writerow([
                            episode_count, update, raw_score, clipped_reward, episode_steps
                        ])
                        episode_csv_file.flush()
                        
                        ep_returns[i] = 0.0
                        ep_lengths[i] = 0

            obs = torch.as_tensor(next_obs_np, device=agent.device)

        # Bootstrap value
        with torch.no_grad():
            _, last_values = agent.net(obs)

        # Get rollout data
        obs_batch = buf.obs
        actions_batch = buf.actions
        rewards_batch = buf.rewards
        dones_batch = buf.dones
        logp_batch = buf.logp
        values_batch = buf.values

        # Compute GAE/returns
        advantages, returns = compute_gae(cfg, rewards_batch, values_batch, dones_batch, last_values)
        
        # Flatten
        T, B = cfg.ROLLOUT_STEPS, cfg.NUM_ENVS
        flat_obs = obs_batch.reshape(T * B, *obs_shape)
        flat_actions = actions_batch.reshape(T * B)
        flat_old_logp = logp_batch.reshape(T * B)
        flat_adv = advantages.reshape(T * B)
        flat_returns = returns.reshape(T * B)
        flat_values = values_batch.reshape(T * B)

        # Normalize advantages
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        # PPO update
        num_samples = T * B
        idxs = np.arange(num_samples)

        policy_losses, value_losses, entropies, clip_fracs, kls = [], [], [], [], []

        for epoch in range(cfg.PPO_EPOCHS):
            np.random.shuffle(idxs)
            for start in range(0, num_samples, cfg.MINIBATCH_SIZE):
                end = start + cfg.MINIBATCH_SIZE
                mb_idx = idxs[start:end]
                mb_obs = flat_obs[mb_idx].to(agent.device)
                mb_actions = flat_actions[mb_idx].to(agent.device)
                mb_old_logp = flat_old_logp[mb_idx].to(agent.device)
                mb_adv = flat_adv[mb_idx].to(agent.device)
                mb_returns = flat_returns[mb_idx].to(agent.device)
                mb_values_old = flat_values[mb_idx].to(agent.device)

                with torch.cuda.amp.autocast(enabled=(cfg.USE_AMP and agent.device.type == "cuda")):
                    new_logp, entropy, new_values = agent.evaluate(mb_obs, mb_actions)
                    ratio = torch.exp(new_logp - mb_old_logp)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - cfg.CLIP_EPS, 1.0 + cfg.CLIP_EPS) * mb_adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    v_pred_clipped = mb_values_old + (new_values - mb_values_old).clamp(-cfg.VALUE_CLIP, cfg.VALUE_CLIP)
                    vf_losses1 = (new_values - mb_returns) ** 2
                    vf_losses2 = (v_pred_clipped - mb_returns) ** 2
                    value_loss = 0.5 * torch.max(vf_losses1, vf_losses2).mean()

                    entropy_loss = -cfg.ENTROPY_COEF * entropy.mean()
                    loss = policy_loss + cfg.VALUE_COEF * value_loss + entropy_loss

                agent.opt.zero_grad(set_to_none=True)
                agent.scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(agent.net.parameters(), cfg.MAX_GRAD_NORM)
                agent.scaler.step(agent.opt)
                agent.scaler.update()

                with torch.no_grad():
                    approx_kl = (mb_old_logp - new_logp).mean().clamp_min(0.0)
                    clipped = (ratio < (1.0 - cfg.CLIP_EPS)) | (ratio > (1.0 + cfg.CLIP_EPS))
                    clip_frac = clipped.float().mean()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
                clip_fracs.append(clip_frac.item())
                kls.append(approx_kl.item())

            # Early stop if KL too large
            recent = max(1, num_samples // cfg.MINIBATCH_SIZE)
            if np.mean(kls[-recent:]) > cfg.TARGET_KL:
                break

        # Logging
        if update % cfg.LOG_INTERVAL == 0:
            window_size = cfg.LOG_INTERVAL * cfg.NUM_ENVS
            reward_window = rewards_history[-window_size:] if len(rewards_history) >= window_size else rewards_history
            score_window = scores_history[-window_size:] if len(scores_history) >= window_size else scores_history
            
            avg_ep_reward = float(np.mean(reward_window)) if reward_window else 0.0
            avg_ep_score = float(np.mean(score_window)) if score_window else 0.0
            
            # Write to CSV
            csv_writer.writerow([
                update,
                len(rewards_history),
                avg_ep_reward,
                avg_ep_score,
                np.mean(policy_losses),
                np.mean(value_losses),
                np.mean(entropies),
                np.mean(clip_fracs),
                np.mean(kls)
            ])
            csv_file.flush()
            
            logger.info(
                f"Update {update}/{cfg.TOTAL_UPDATES} | "
                f"LR: {lrnow:.2e} | "
                f"Avg Reward: {avg_ep_reward:.1f} | "
                f"Avg Score: {avg_ep_score:.1f} | "
                f"Policy Loss: {np.mean(policy_losses):.4f} | "
                f"Value Loss: {np.mean(value_losses):.4f} | "
                f"Entropy: {np.mean(entropies):.3f} | "
                f"KL: {np.mean(kls):.4f}"
            )

        if update % cfg.SAVE_INTERVAL == 0:
            path = os.path.join(cfg.MODEL_DIR, f"agent_update_{update}.pt")
            torch.save({
                "model": agent.net.state_dict(),
                "optimizer": agent.opt.state_dict(),
                "scaler": agent.scaler.state_dict(),
                "update": update,
                "rewards_history": rewards_history,
                "scores_history": scores_history,
                "episode_count": episode_count,
                "cfg": cfg.__dict__
            }, path)
            logger.info(f"Saved checkpoint to {path}")
            
            # Plot rewards at same interval as saving
            plot_rewards(rewards_history, cfg.LOG_DIR, update, cfg.TOTAL_UPDATES, is_final=False)

    # Final save
    final_path = os.path.join(cfg.MODEL_DIR, "agent_final.pt")
    torch.save({
        "model": agent.net.state_dict(),
        "optimizer": agent.opt.state_dict(),
        "scaler": agent.scaler.state_dict(),
        "update": cfg.TOTAL_UPDATES,
        "rewards_history": rewards_history,
        "scores_history": scores_history,
        "episode_count": episode_count,
        "cfg": cfg.__dict__
    }, final_path)
    logger.info(f"Saved final model to {final_path}")

    # Final plot
    plot_rewards(rewards_history, cfg.LOG_DIR, cfg.TOTAL_UPDATES, cfg.TOTAL_UPDATES, is_final=True)

    # Close CSV files
    csv_file.close()
    episode_csv_file.close()
    logger.info(f"Training logs saved to {csv_path} and {episode_csv_path}")

    env.close()
    return final_path

@torch.no_grad()
def play(cfg: PPOConfig, model_path: str, episodes: int = 5):
    env = make_single_env(cfg, render=True)()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_dim = env.action_space.n

    net = ACNetCNN(action_dim).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint["model"])
    net.eval()

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        raw_score = 0.0
        steps = 0
        while not done:
            obs_t = torch.as_tensor(obs[None, ...], device=device)
            logits, _ = net(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample().item()
            obs, r, terminated, truncated, info = env.step(action)
            total_reward += r  # This is the scaled reward now
            steps += 1
            done = terminated or truncated
            
            # Update raw score from info (updated every step)
            if 'raw_score' in info:
                raw_score = info['raw_score']
        
        logger.info(f"Episode {ep+1}: Scaled Reward={total_reward:.1f}, Raw Score={raw_score:.1f}, Steps={steps}")
    env.close()

def train_ppo_agent(agent_class, config: PPOConfig, render_human: bool = False):
    logger.info("Starting PPO training...")
    if render_human:
        logger.warning("render_human=True disabled for PPO training, it's way too slow.")
    
    final_model_path = train(cfg=config, resume_from=None)
    return final_model_path


def play_ppo_agent(agent_class, config: PPOConfig, model_path: str, num_episodes: int = 5):
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    logger.info(f"Playing {num_episodes} episodes with trained PPO agent from {model_path}")
    play(cfg=config, model_path=model_path, episodes=num_episodes)