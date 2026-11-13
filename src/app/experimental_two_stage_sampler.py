"""W139 (EXPERIMENTAL, opt-in, scratch only): scheduled-replacement sampler for UniMate Stage 2.

This is a NON-OFFICIAL sampler procedure built on UniMate's OFFICIAL replacement primitive.  It is a verbatim copy of
``unimate.inference.motion_inbetweening.inbetween_sample_ode`` (pinned code ``b78c7807``, file sha256
``ec02743b...``) with exactly ONE behavioural change: the keep mask is chosen per produced state.

    official:   x_j = where(keep_mask,                         (1 - t_j) eps + t_j x1_known, x_j)   for every state j
    scheduled:  x_j = where(keep_mask | (sched_mask & j <= k_s), (1 - t_j) eps + t_j x1_known, x_j)

State indexing (integer, never a float comparison on t): state 0 is produced by the initial replacement at ``ts[0]``;
state ``i + 1`` is produced by the replacement after Euler step ``i`` (``_replace(x, ts[i + 1])``).  Therefore

  * ``k_s = 0``: state 0 on ``sched_mask`` is ``1.0 * eps + 0.0 * x1 = eps`` (bit-identical to the free initial state),
    so the procedure is behaviourally identical to the official function;
  * ``k_s = K``: states 0..K are fully pinned on ``keep_mask | sched_mask``; the first network evaluation whose output
    is NOT overwritten on ``sched_mask`` is Euler step ``K`` at ``t = ts[K]``, whose input on ``sched_mask`` is exactly
    the flow-matching training marginal ``(1 - t_K) eps + t_K x1`` (linear ICPlan path, data at t = 1).  This is the
    flow form of SDEdit (Meng et al., ICLR 2022): Stage-1 content sets the ODE state at ``t_s = ts[K]`` and the
    network integrates the rest freely.  Network outputs of steps ``0..K-1`` on ``sched_mask`` are computed and
    discarded (overwritten);
  * ``k_s = num_steps``: every state is pinned -> the output equals ``x1_known`` on ``keep_mask | sched_mask``.

The model weights, the CFG wrapper, the transport interval, the step count and the noise draw are untouched.  The
optional ``recorder`` only copies tensors to CPU; it never changes a value.  Imports torch only (no unimate import), so
synthetic tests can compare it against the pinned official function with a stub velocity model.
"""
from __future__ import annotations

import hashlib

import torch


def tensor_sha256(t):
    a = t.detach().to("cpu").contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


class StateRecorder:
    """Read-only recorder of the sampler trajectory (valid joints only)."""

    def __init__(self, n_valid, k_s):
        self.n_valid = int(n_valid)
        self.k_s = int(k_s)
        self.states, self.velocities, self.t, self.mask_sha = [], [], [], []
        self.eps = None
        self.eps_sha256 = None
        self.calls = 0
        self.first_free_step_input = None

    def on_eps(self, eps):
        self.eps = eps.detach().to("cpu")[:, :self.n_valid].clone()
        self.eps_sha256 = tensor_sha256(eps)

    def on_state(self, j, x, t, mask):
        self.states.append(x.detach().to("cpu")[:, :self.n_valid].clone())
        self.t.append(float(t))
        self.mask_sha.append(tensor_sha256(mask.to(torch.uint8)))

    def on_velocity(self, i, x_in, v):
        self.velocities.append(v.detach().to("cpu")[:, :self.n_valid].clone())
        if i == self.k_s:
            self.first_free_step_input = x_in.detach().to("cpu")[:, :self.n_valid].clone()

    def arrays(self):
        return {"states": torch.stack(self.states).numpy(), "velocities": torch.stack(self.velocities).numpy(),
                "t": torch.tensor(self.t, dtype=torch.float64).numpy(), "eps": self.eps.numpy(),
                "first_free_step_input": (self.first_free_step_input.numpy() if self.first_free_step_input is not None else None)}


def scheduled_inbetween_sample_ode(
    sample_model,
    transport,
    cond: dict,
    x1_known: torch.Tensor,
    keep_mask: torch.Tensor,
    motion_shape,
    *,
    num_steps: int = 50,
    device: torch.device,
    sched_mask: torch.Tensor = None,
    k_s: int = 0,
    recorder: StateRecorder = None,
) -> torch.Tensor:
    """Verbatim official ``inbetween_sample_ode`` + the per-state mask ``keep_mask | (sched_mask & j <= k_s)``."""
    if not isinstance(k_s, int) or isinstance(k_s, bool) or not 0 <= k_s <= num_steps:
        raise ValueError(f"k_s must be an int in [0, {num_steps}], got {k_s!r}")
    t0, t1 = transport.check_interval(
        transport.train_eps,
        transport.sample_eps,
        sde=False, eval=True, reverse=False, last_step_size=0.0,
    )

    eps = torch.randn(motion_shape, device=device)
    x = eps.clone()
    if recorder is not None:
        recorder.calls += 1
        recorder.on_eps(eps)

    base_mask = keep_mask
    if sched_mask is None:
        sched = torch.zeros(tuple(motion_shape), dtype=torch.bool, device=keep_mask.device)
    else:
        if sched_mask.dtype != torch.bool:
            raise ValueError("sched_mask must be a bool tensor")
        sched = sched_mask
    widened = torch.broadcast_to(base_mask, motion_shape) | torch.broadcast_to(sched, motion_shape)

    def _mask_for_state(j):
        return widened if j <= k_s else base_mask

    def _replace(x_t, t_scalar, j):
        mask = _mask_for_state(j)
        t_blend = (1.0 - t_scalar) * eps + t_scalar * x1_known
        out = torch.where(mask, t_blend, x_t)
        if recorder is not None:
            recorder.on_state(j, out, t_scalar, torch.broadcast_to(mask, motion_shape))
        return out

    x = _replace(x, t0, 0)

    ts = torch.linspace(t0, t1, num_steps + 1, device=device)

    for i in range(num_steps):
        t_cur = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t_cur

        t_batch = torch.full((x.size(0),), t_cur.item(), device=device, dtype=x.dtype)
        v = sample_model(x, t_batch, cond=cond)
        if recorder is not None:
            recorder.on_velocity(i, x, v)
        x = x + dt * v
        x = _replace(x, t_next, i + 1)

    return x
