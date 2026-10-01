"""Demo dataset loading + sampling for BC pretraining and DAPG.

The ``scripted_ik`` CLI (or any compatible producer) writes demo
rollouts as a single compressed ``.npz`` file with the schema::

    obs                 : (T, obs_dim)    float32   policy-group observations
    next_obs            : (T, obs_dim)    float32   policy-group observations
    actions             : (T, act_dim)    float32   normalized actions in [-1, 1]
    dones               : (T,)            bool      per-transition done flags
    episode_ids         : (T,)            int32     episode index each transition belongs to
    presets             : (n_ep,)         object    preset name per episode
    successes           : (n_ep,)         bool      scripted success flag per episode
    max_cube_heights    : (n_ep,)         float32
    final_cube_heights  : (n_ep,)         float32
    cube_start          : (n_ep, 3)       float32
    goal_start          : (n_ep, 3)       float32
    steps_per_episode   : (n_ep,)         int32

The :class:`DemoBuffer` wraps this file as a PyTorch-friendly replay:
it lazy-loads into device memory once, optionally filters to
successful episodes only, and exposes a simple :meth:`sample` method
that returns ``(states, actions)`` tensors suitable for both BC
pretraining (see :mod:`skrl.utils.bc`) and PPO+BC (DAPG) loss mixing
(see :class:`skrl.agents.torch.ppo.dapg.PPO_DAPG`).

:class:`TransitionDemoBuffer` extends the schema to include rewards,
next-states, and termination/truncation flags so it can feed RLPD-style
offline-replay SAC updates (see
:class:`skrl.agents.torch.sac.rlpd.RLPD`).

Usage::

    from skrl.memories.torch.demo import DemoBuffer

    buf = DemoBuffer.from_npz(
        "demos/sudo_lift_cube.npz", device="cuda:0", only_successful=True
    )
    states, actions = buf.sample(batch_size=256)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

_log = logging.getLogger(__name__)


@dataclass
class DemoMetadata:
    """Lightweight summary attached to every :class:`DemoBuffer`."""

    path: str
    num_episodes: int
    num_transitions: int
    obs_dim: int
    action_dim: int
    successful_episodes: int
    presets: tuple[str, ...]
    filtered_only_successful: bool


class DemoBuffer:
    """In-memory demo-transition store with a uniform random sampler.

    Filtering behaviour:
        * ``only_successful=True`` drops any transition whose episode has
          ``successes[ep]==False``.  This is the default for DAPG / BC
          because training on failed imitation trajectories biases the
          policy toward the same failure modes.
        * ``preset_allowlist`` further filters by preset name.  Passing
          ``None`` (default) keeps every preset.

    After filtering, ``states`` / ``actions`` tensors live on ``device``
    and are reused for every :meth:`sample` call.  Keeping them resident
    avoids per-sample CPU→GPU copies inside tight PPO update loops.
    """

    def __init__(
        self,
        *,
        states: torch.Tensor,
        actions: torch.Tensor,
        metadata: DemoMetadata,
        device: str | torch.device,
    ) -> None:
        if states.shape[0] != actions.shape[0]:
            raise ValueError(f"states/actions transition count mismatch: {states.shape[0]} vs {actions.shape[0]}")
        self.device = torch.device(device)
        self.states = states.to(self.device)
        self.actions = actions.to(self.device)
        self.metadata = metadata

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        only_successful: bool = True,
        preset_allowlist: tuple[str, ...] | None = None,
    ) -> "DemoBuffer":
        """Load an ``.npz`` dataset produced by a scripted demo collector."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Demo dataset not found: {path}")

        with np.load(path, allow_pickle=True) as data:
            obs = np.asarray(data["obs"], dtype=np.float32)
            actions = np.asarray(data["actions"], dtype=np.float32)
            episode_ids = np.asarray(data["episode_ids"], dtype=np.int64)
            presets = np.asarray(data["presets"], dtype=object)
            successes = np.asarray(data["successes"], dtype=bool)

        keep = np.ones(obs.shape[0], dtype=bool)
        if only_successful:
            per_trans_success = successes[episode_ids]
            keep &= per_trans_success
        if preset_allowlist is not None:
            allow = set(preset_allowlist)
            per_trans_preset = presets[episode_ids]
            mask = np.array([p in allow for p in per_trans_preset], dtype=bool)
            keep &= mask

        if not keep.any():
            raise ValueError(
                f"No demo transitions passed the filters "
                f"(only_successful={only_successful}, presets={preset_allowlist})."
                f"  Source file has {obs.shape[0]} transitions across "
                f"{int(successes.sum())}/{len(successes)} successful episodes."
            )

        kept_obs = obs[keep]
        kept_actions = actions[keep]

        metadata = DemoMetadata(
            path=str(path),
            num_episodes=int(len(successes)),
            num_transitions=int(kept_obs.shape[0]),
            obs_dim=int(kept_obs.shape[1]),
            action_dim=int(kept_actions.shape[1]),
            successful_episodes=int(successes.sum()),
            presets=tuple(sorted({str(p) for p in presets})),
            filtered_only_successful=bool(only_successful),
        )
        _log.info(
            "Loaded demo buffer: %d transitions across %d episodes (%d successful); "
            "obs_dim=%d, action_dim=%d, presets=%s",
            metadata.num_transitions,
            metadata.num_episodes,
            metadata.successful_episodes,
            metadata.obs_dim,
            metadata.action_dim,
            metadata.presets,
        )

        states_t = torch.from_numpy(kept_obs)
        actions_t = torch.from_numpy(kept_actions)
        return cls(
            states=states_t,
            actions=actions_t,
            metadata=metadata,
            device=device,
        )

    def __len__(self) -> int:
        return int(self.states.shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.states.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    def sample(self, batch_size: int, *, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniformly sample ``batch_size`` transitions.

        Returns:
          ``(states, actions)`` tensors on ``self.device``.  Shapes are
          ``(batch_size, obs_dim)`` and ``(batch_size, action_dim)``.
        """
        n = len(self)
        if n == 0:
            raise RuntimeError("DemoBuffer is empty")
        idx = torch.randint(
            low=0,
            high=n,
            size=(batch_size,),
            device=self.device,
            generator=generator,
        )
        return self.states[idx], self.actions[idx]

    def iter_minibatches(self, batch_size: int, *, shuffle: bool = True) -> Any:
        """Yield ``(states, actions)`` minibatches covering the whole buffer."""
        n = len(self)
        if shuffle:
            perm = torch.randperm(n, device=self.device)
        else:
            perm = torch.arange(n, device=self.device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            yield self.states[idx], self.actions[idx]

    def describe(self) -> str:
        m = self.metadata
        return (
            f"DemoBuffer('{m.path}'): {m.num_transitions} transitions | "
            f"{m.num_episodes} episodes ({m.successful_episodes} success) | "
            f"obs_dim={m.obs_dim} action_dim={m.action_dim} | "
            f"presets={m.presets} only_successful={m.filtered_only_successful}"
        )


@dataclass
class TransitionBatch:
    """Batch sampled from an offline transition replay."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class TransitionDemoBuffer(DemoBuffer):
    """Offline replay buffer for RLPD-style SAC updates."""

    def __init__(
        self,
        *,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        metadata: DemoMetadata,
        device: str | torch.device,
    ) -> None:
        super().__init__(states=states, actions=actions, metadata=metadata, device=device)
        self.rewards = rewards.to(self.device)
        self.next_states = next_states.to(self.device)
        self.terminated = terminated.to(self.device)
        self.truncated = truncated.to(self.device)

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        only_successful: bool = True,
        preset_allowlist: tuple[str, ...] | None = None,
    ) -> "TransitionDemoBuffer":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Demo dataset not found: {path}")

        with np.load(path, allow_pickle=True) as data:
            required = ("obs", "next_obs", "actions", "rewards", "dones", "episode_ids", "presets", "successes")
            missing = [name for name in required if name not in data]
            if missing:
                raise KeyError(
                    f"RLPD replay file {path} is missing {missing}. "
                    "Regenerate demos with rewards/next_obs/dones enabled."
                )
            obs = np.asarray(data["obs"], dtype=np.float32)
            next_obs = np.asarray(data["next_obs"], dtype=np.float32)
            actions = np.asarray(data["actions"], dtype=np.float32)
            rewards = np.asarray(data["rewards"], dtype=np.float32).reshape(-1, 1)
            dones = np.asarray(data["dones"], dtype=bool).reshape(-1, 1)
            episode_ids = np.asarray(data["episode_ids"], dtype=np.int64)
            presets = np.asarray(data["presets"], dtype=object)
            successes = np.asarray(data["successes"], dtype=bool)

        keep = np.ones(obs.shape[0], dtype=bool)
        if only_successful:
            keep &= successes[episode_ids]
        if preset_allowlist is not None:
            allow = set(preset_allowlist)
            per_trans_preset = presets[episode_ids]
            keep &= np.array([p in allow for p in per_trans_preset], dtype=bool)
        if not keep.any():
            raise ValueError(f"No RLPD transitions passed filters for {path}")

        kept_obs = obs[keep]
        kept_actions = actions[keep]
        metadata = DemoMetadata(
            path=str(path),
            num_episodes=int(len(successes)),
            num_transitions=int(kept_obs.shape[0]),
            obs_dim=int(kept_obs.shape[1]),
            action_dim=int(kept_actions.shape[1]),
            successful_episodes=int(successes.sum()),
            presets=tuple(sorted({str(p) for p in presets})),
            filtered_only_successful=bool(only_successful),
        )
        return cls(
            states=torch.from_numpy(kept_obs),
            actions=torch.from_numpy(kept_actions),
            rewards=torch.from_numpy(rewards[keep]),
            next_states=torch.from_numpy(next_obs[keep]),
            terminated=torch.from_numpy(dones[keep]),
            truncated=torch.zeros((int(keep.sum()), 1), dtype=torch.bool),
            metadata=metadata,
            device=device,
        )

    def sample_transitions(self, batch_size: int) -> TransitionBatch:
        n = len(self)
        idx = torch.randint(0, n, (batch_size,), device=self.device)
        return TransitionBatch(
            states=self.states[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_states=self.next_states[idx],
            terminated=self.terminated[idx],
            truncated=self.truncated[idx],
        )


__all__ = [
    "DemoBuffer",
    "DemoMetadata",
    "TransitionBatch",
    "TransitionDemoBuffer",
]
