"""Behavior-cloning pretraining utilities for DAPG / warm-start PPO.

This module provides a small, algorithm-agnostic BC loop that operates
on whatever actor model the downstream RL agent uses.  The policy is
expected to follow the skrl / rlmodule contract — i.e.
``policy.act({"states": states, "taken_actions": actions},
role="policy")`` returns a ``(sampled, log_prob, outputs)`` triple with
``outputs["mean_actions"]``.

Two loss flavours are supported:

* ``"nll"`` (default) — negative log-likelihood of the demo action
  under the current Gaussian.  Matches the DAPG auxiliary loss and
  behaves well when action sigmas are learnable.
* ``"mse"`` — mean-squared-error between the demo action and the
  policy *mean*.  Useful as a deterministic baseline and as a sanity
  check that the network can overfit the demo set.

The function returns a compact training summary so callers can log
pre-training statistics (final loss, action MSE on a held-out batch).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn.functional as F

from skrl.memories.torch.demo import DemoBuffer

_log = logging.getLogger(__name__)


@dataclass
class BCPretrainResult:
    """Summary returned from :func:`pretrain_bc`."""

    epochs: int
    batches_per_epoch: int
    final_loss: float
    final_mse_on_mean: float
    wall_time_s: float
    loss_history: list[float] = field(default_factory=list)


def pretrain_bc(
    *,
    policy: torch.nn.Module,
    demo_buffer: DemoBuffer,
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    loss_type: Literal["nll", "mse"] = "nll",
    state_preprocessor: Any = None,
    log_every: int = 50,
) -> BCPretrainResult:
    """Fit ``policy`` to ``demo_buffer`` for ``epochs`` full passes.

    Args:
      policy: skrl / rlmodule policy model.  Must expose ``.act({"states":
        ..., "taken_actions": ...}, role="policy")``.  We do NOT touch the
        value head; the caller is responsible for value warm-up if needed.
      demo_buffer: In-memory :class:`DemoBuffer`.
      epochs: Number of full passes over the demo set.
      batch_size: Per-step minibatch size.
      learning_rate: Adam learning rate.
      loss_type: ``"nll"`` (recommended for downstream DAPG) or ``"mse"``.
      state_preprocessor: Optional skrl ``RunningStandardScaler`` (or any
        object with a ``__call__(states, train=bool)`` signature).  If
        given, we normalise states before feeding the policy, matching
        the PPO runtime pipeline.  We pass ``train=True`` so the
        preprocessor's running stats warm up on demo data *before* PPO
        starts interacting.
      log_every: Print a summary every ``log_every`` batches.

    Returns:
      :class:`BCPretrainResult` with the final metrics and loss trace.
    """
    if epochs <= 0:
        _log.info("BC pretraining skipped (epochs=%d)", epochs)
        return BCPretrainResult(
            epochs=0,
            batches_per_epoch=0,
            final_loss=0.0,
            final_mse_on_mean=float("nan"),
            wall_time_s=0.0,
        )

    policy.train()
    optim = torch.optim.Adam(policy.parameters(), lr=float(learning_rate))

    n_trans = len(demo_buffer)
    batches_per_epoch = max(1, n_trans // batch_size)
    loss_history: list[float] = []
    last_loss = float("nan")
    last_mse = float("nan")

    _log.info(
        "BC pretraining: %d epochs x %d batches/epoch (bs=%d, lr=%.2e, loss=%s)",
        epochs,
        batches_per_epoch,
        batch_size,
        learning_rate,
        loss_type,
    )
    print(
        f"[BC] pretraining start: epochs={epochs} batches_per_epoch={batches_per_epoch} "
        f"batch_size={batch_size} lr={learning_rate:.2e} loss={loss_type}",
        flush=True,
    )
    t0 = time.perf_counter()

    for epoch in range(epochs):
        for batch_i, (states, actions) in enumerate(demo_buffer.iter_minibatches(batch_size=batch_size, shuffle=True)):
            if states.shape[0] < 2:
                continue

            if state_preprocessor is not None:
                states_pp = state_preprocessor(states, train=True)
            else:
                states_pp = states

            sampled_actions, log_prob, outputs = policy.act(
                {"states": states_pp, "taken_actions": actions}, role="policy"
            )
            mean_actions = outputs.get("mean_actions", sampled_actions)

            if loss_type == "nll":
                loss = -log_prob.mean()
            elif loss_type == "mse":
                loss = F.mse_loss(mean_actions, actions)
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            optim.zero_grad()
            loss.backward()
            optim.step()

            with torch.no_grad():
                mse = F.mse_loss(mean_actions, actions).item()

            last_loss = float(loss.item())
            last_mse = float(mse)
            loss_history.append(last_loss)

            global_batch_i = epoch * batches_per_epoch + batch_i
            if log_every > 0 and (global_batch_i % log_every == 0):
                _log.info(
                    "  BC epoch %d/%d batch %d loss=%.4f mse=%.4f",
                    epoch + 1,
                    epochs,
                    batch_i,
                    last_loss,
                    last_mse,
                )

    wall = time.perf_counter() - t0
    _log.info(
        "BC pretraining done in %.1fs.  final_loss=%.4f final_mse=%.4f",
        wall,
        last_loss,
        last_mse,
    )
    print(
        f"[BC] pretraining done: wall_time_s={wall:.1f} final_loss={last_loss:.6f} final_mse={last_mse:.6f}",
        flush=True,
    )

    return BCPretrainResult(
        epochs=epochs,
        batches_per_epoch=batches_per_epoch,
        final_loss=last_loss,
        final_mse_on_mean=last_mse,
        wall_time_s=wall,
        loss_history=loss_history,
    )


__all__ = ["BCPretrainResult", "pretrain_bc"]
