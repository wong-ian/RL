# deep_ql_agent.py
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple
from config.base import BaseConfig

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -------- BaseAgent compatibility --------
try:
    # Use the project's BaseAgent if available
    from agents.base import BaseAgent  # type: ignore
except Exception:
    # Fallback for running this file standalone
    class BaseAgent:
        def __init__(self, action_space, config=None): pass
        def get_action(self, observation): ...
        def update(self, state, action, reward, next_state, needs_to_reset): ...
        def save(self, filepath): ...
        def load(self, filepath): ...

# ====================================================
# Utilities
# ====================================================

def chw_from_obs(obs: np.ndarray) -> torch.Tensor:
    """
    Convert an observation to a torch.FloatTensor.
    - Vector (1D): returns shape [D]
    - 2D grayscale: returns [1, H, W]
    - Image HWC or CHW: returns [C, H, W]
    Scales to [0,1] if the max value > 1 (i.e., uint8 images).
    """
    if obs is None:
        raise ValueError("Observation is None")
    arr = np.asarray(obs)
    if arr.ndim == 1:  # vector
        t = torch.as_tensor(arr, dtype=torch.float32)
        return t
    if arr.ndim == 2:  # grayscale image -> 1xHxW
        t = torch.as_tensor(arr, dtype=torch.float32).unsqueeze(0)
        return t / 255.0 if t.max() > 1.0 else t
    if arr.ndim == 3:
        # HWC or CHW?
        if arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[-1]:
            # Looks like CHW already (C,H,W) with small C
            t = torch.as_tensor(arr, dtype=torch.float32)
        else:
            # Assume HWC -> CHW
            t = torch.as_tensor(arr, dtype=torch.float32).permute(2, 0, 1)
        return t / 255.0 if t.max() > 1.0 else t
    raise ValueError(f"Unsupported obs shape: {arr.shape}")


# ====================================================
# Q-Networks
# ====================================================

class CNNQ(nn.Module):
    def __init__(self, in_channels: int, n_actions: int, image_size: Tuple[int, int] | None = None):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            if image_size is None:
                dummy = torch.zeros(1, in_channels, 84, 84)
            else:
                dummy = torch.zeros(1, in_channels, image_size[0], image_size[1])
            n_flat = self.conv(dummy).view(1, -1).size(1)
        self.head = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.head(x)


class MLPQ(nn.Module):
    def __init__(self, in_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.net(x)


# ====================================================
# Replay Buffer
# ====================================================

class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.idx = 0
        self.full = False

        self.s = [None] * capacity
        self.a = np.zeros((capacity,), dtype=np.int64)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.s2 = [None] * capacity
        self.d = np.zeros((capacity,), dtype=np.bool_)

    def push(self, s, a, r, s2, done):
        i = self.idx
        self.s[i] = np.array(s, copy=False)
        self.a[i] = int(a)
        self.r[i] = float(r)
        self.s2[i] = np.array(s2, copy=False)
        self.d[i] = bool(done)
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample(self, batch_size: int):
        n = len(self)
        idxs = np.random.randint(0, n, size=batch_size, dtype=np.int32)
        s  = torch.stack([chw_from_obs(self.s[i]) for i in idxs]).to(self.device)
        s2 = torch.stack([chw_from_obs(self.s2[i]) for i in idxs]).to(self.device)
        a  = torch.as_tensor(self.a[idxs], device=self.device, dtype=torch.long)
        r  = torch.as_tensor(self.r[idxs], device=self.device, dtype=torch.float32)
        d  = torch.as_tensor(self.d[idxs], device=self.device, dtype=torch.float32)
        return s, a, r, s2, d


# ====================================================
# Config + Agent
# ====================================================

@dataclass
class DQNConfig(BaseConfig):
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 64
    buffer_size: int = 100_000
    learning_starts: int = 5_000
    train_freq: int = 4
    target_update_freq: int = 2_000
    eps_start: float = 1.0
    eps_final: float = 0.05
    eps_decay_steps: int = 200_000
    max_grad_norm: float = 10.0
    double_dqn: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Optional hints for images
    is_image: Optional[bool] = None
    image_size: Optional[Tuple[int, int]] = None  # (H, W) if known


class DQNAgent(BaseAgent):

    """
    Drop-in DQN that matches your BaseAgent API:
      - get_action(observation) -> int
      - update(state, action, reward, next_state, needs_to_reset) -> dict
      - save(filepath), load(filepath)

    Works with vector or image observations.
    """
    def __init__(self, action_space, config=None):
        super().__init__(action_space, config)
        self.action_space = action_space
        self.cfg = config
        self.device = torch.device(self.cfg.device)

        # Lazy init after first observation
        self.q: Optional[nn.Module] = None
        self.q_target: Optional[nn.Module] = None
        self.opt: Optional[optim.Optimizer] = None
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.buffer: Optional[ReplayBuffer] = None

        self.step_count = 0
        self._pending_ckpt = None  # for load-before-init

    # ---------- BaseAgent API ----------
    def get_action(self, observation):
        if self.q is None:
            self._init_from_obs(observation)

        eps = self._epsilon()
        self.step_count += 1
        if random.random() < eps:
            return self.action_space.sample()

        x = chw_from_obs(observation).to(self.device)
        with torch.no_grad():
            q = self.q(x)
            a = int(torch.argmax(q, dim=1).item())
        return a

    def update(self, state, action, reward, next_state, needs_to_reset):
        if self.buffer is None:
            self._init_from_obs(state)

        done = bool(needs_to_reset)
        self.buffer.push(state, action, reward, next_state, done)

        # Warmup
        if len(self.buffer) < self.cfg.learning_starts:
            return {}

        # Train on schedule
        if self.step_count % self.cfg.train_freq != 0:
            return {}

        s, a, r, s2, d = self.buffer.sample(self.cfg.batch_size)

        with torch.no_grad():
            if self.cfg.double_dqn:
                a2 = torch.argmax(self.q(s2), dim=1)
                q_next = self.q_target(s2).gather(1, a2.unsqueeze(1)).squeeze(1)
            else:
                q_next = self.q_target(s2).max(1).values
            target = r + (1.0 - d) * self.cfg.gamma * q_next

        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(q_sa, target).mean()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.max_grad_norm)
        self.opt.step()

        # Target sync
        if self.step_count % self.cfg.target_update_freq == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return {"loss": float(loss.item()), "epsilon": self._epsilon(), "buffer": len(self.buffer)}

    def save(self, filepath: str):
        if self.q is None:
            raise RuntimeError("Agent not initialized; call get_action() once before save.")
        torch.save({
            "model": self.q.state_dict(),
            "target": self.q_target.state_dict(),
            "opt": self.opt.state_dict(),
            "cfg": self.cfg.__dict__,
            "step_count": self.step_count,
        }, filepath)

    def load(self, filepath: str):
        ckpt = torch.load(filepath, map_location=self.device)
        # merge cfg (preserve current defaults where not present)
        self.cfg = DQNConfig(**{**self.cfg.__dict__, **ckpt.get("cfg", {})})
        self.step_count = ckpt.get("step_count", 0)
        # If models aren't built yet, defer restoration
        self._pending_ckpt = ckpt
        if self.q is not None:
            self._restore_from_ckpt()

    # ---------- Internals ----------
    def _epsilon(self):
        c = self.cfg
        if self.step_count >= c.eps_decay_steps:
            return c.eps_final
        frac = self.step_count / c.eps_decay_steps
        return c.eps_start + frac * (c.eps_final - c.eps_start)

    def _init_from_obs(self, obs):
        x = chw_from_obs(obs).to(self.device)
        # Decide image vs vector
        is_image = self.cfg.is_image
        if is_image is None:
            is_image = (x.ndim == 3 and min(x.shape[-2:]) > 1)

        if is_image:
            c, h, w = x.shape[-3], x.shape[-2], x.shape[-1]
            self.q = CNNQ(c, self.action_space.n, image_size=(h, w)).to(self.device)
            self.q_target = CNNQ(c, self.action_space.n, image_size=(h, w)).to(self.device)
        else:
            dim = int(x.numel())
            self.q = MLPQ(dim, self.action_space.n).to(self.device)
            self.q_target = MLPQ(dim, self.action_space.n).to(self.device)

        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()
        self.opt = optim.AdamW(self.q.parameters(), lr=self.cfg.lr, amsgrad=True)
        self.buffer = ReplayBuffer(self.cfg.buffer_size, self.device)

        # If a checkpoint was loaded before shapes were known, restore now
        if self._pending_ckpt is not None:
            self._restore_from_ckpt()

    def _restore_from_ckpt(self):
        ckpt = self._pending_ckpt
        if ckpt is None:
            return
        try:
            self.q.load_state_dict(ckpt["model"])
            self.q_target.load_state_dict(ckpt["target"])
            self.opt.load_state_dict(ckpt["opt"])
        finally:
            self._pending_ckpt = None


# ====================================================
# Optional CLI for quick testing
# ====================================================

def _run_cli():
    """
    Minimal training loop for sanity checks.
    Usage:
      python -m agents.deep_ql_agent --env CartPole-v1 --steps 10000
      python -m agents.deep_ql_agent --env ALE/MsPacman-v5 --steps 20000
    """
    import gymnasium as gym

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--steps", type=int, default=10000)
    args = parser.parse_args()

    env = gym.make(args.env)
    obs, _ = env.reset()

    agent = DQNAgent(env.action_space, DQNConfig())
    ep_ret, ep_len = 0.0, 0

    for t in range(1, args.steps + 1):
        a = agent.get_action(obs)
        nxt, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        metrics = agent.update(obs, a, r, nxt, done)
        obs = nxt
        ep_ret += r
        ep_len += 1

        if done:
            print(f"[episode] return={ep_ret:.2f} length={ep_len} steps={t}")
            obs, _ = env.reset()
            ep_ret, ep_len = 0.0, 0

        if t % 1000 == 0:
            print(f"[t={t}] eps={metrics.get('epsilon') if metrics else None} "
                  f"buffer={metrics.get('buffer') if metrics else len(agent.buffer) if agent.buffer else 0} "
                  f"loss={metrics.get('loss') if metrics else None}")

    env.close()
    print("Done.")

if __name__ == "__main__":
    _run_cli()
