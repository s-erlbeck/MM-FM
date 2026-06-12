from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _as_tuple(values: Any, length: int = 2) -> tuple:
    if isinstance(values, (list, tuple)):
        if len(values) != length:
            raise ValueError(f"Expected sequence of length {length}, got {len(values)}.")
        return tuple(float(v) for v in values)
    return tuple(float(values) for _ in range(length))


def build_optimizer(parameters: Iterable[torch.nn.Parameter], training_cfg: Dict[str, Any]) -> tuple[Optimizer, str]:
    """
    Initialize the optimizer from config. Defaults to AdamW with the legacy base_lr.
    """
    opt_cfg: Dict[str, Any] = dict(training_cfg.get("optimizer", {}))
    opt_type = opt_cfg.get("type", "adamw").lower()
    if opt_type != "adamw":
        raise ValueError(f"Unsupported optimizer '{opt_type}'. Only AdamW is currently available.")

    base_lr = float(opt_cfg.get("lr", training_cfg.get("base_lr", 2e-4)))
    betas = _as_tuple(opt_cfg.get("betas", opt_cfg.get("beta", (0.9, 0.95))))
    weight_decay = float(opt_cfg.get("weight_decay", opt_cfg.get("wd", 0.0)))
    eps = float(opt_cfg.get("eps", 1e-8))

    optimizer = torch.optim.AdamW(
        parameters,
        lr=base_lr,
        betas=betas,
        weight_decay=weight_decay,
        eps=eps,
    )

    training_cfg.setdefault("base_lr", base_lr)
    training_cfg.setdefault("final_lr", float(training_cfg.get("final_lr", base_lr)))
    optim_msg = f"Optimizer: AdamW with lr={base_lr}, betas={betas}, weight_decay={weight_decay}, eps={eps}"
    return optimizer, optim_msg


def build_scheduler(
    optimizer: Optimizer,
    steps_per_epoch: int,
    training_cfg: Dict[str, Any],
    state_dict: Optional[Dict[str, Any]] = None,
) -> tuple[LambdaLR, str]:
    """
    Create a learning rate scheduler with optional initial phase and warmup.
    Supports 'linear' and 'cosine' decay schedules.

    Schedule phases:
    1. Initial phase (optional): constant initial_lr for initial_phase_epochs
    2. Warmup phase: constant base_lr from initial_phase_epochs to decay_start_epoch
    3. Decay phase: decay from base_lr to final_lr (linear or cosine)
    4. Post-decay: constant final_lr
    """
    sched_cfg: Dict[str, Any] = dict(training_cfg.get("scheduler", {}))
    schedule_type = sched_cfg.get("type", training_cfg.get("schedule_type", "linear")).lower()

    base_lr = float(sched_cfg.get("base_lr", training_cfg.get("base_lr", optimizer.param_groups[0]["lr"])))
    final_lr = float(sched_cfg.get("final_lr", training_cfg.get("final_lr", base_lr)))
    final_ratio = final_lr / base_lr if base_lr > 0 else 1.0

    # Initial phase configuration (optional)
    initial_phase_epochs = float(sched_cfg.get("initial_phase_epochs", training_cfg.get("initial_phase_epochs", 0)))
    initial_phase_steps = int(initial_phase_epochs * steps_per_epoch)
    initial_lr = float(sched_cfg.get("initial_lr", training_cfg.get("initial_lr", base_lr)))
    initial_ratio = initial_lr / base_lr if base_lr > 0 else 1.0

    # Warmup/constant phase ends at decay_start_epoch
    warmup_steps_cfg = sched_cfg.get("warmup_steps")
    if warmup_steps_cfg is not None:
        warmup_steps = int(warmup_steps_cfg)
    else:
        warmup_epochs = float(sched_cfg.get("warmup_epochs", training_cfg.get("decay_start_epoch", 0)))
        warmup_steps = int(warmup_epochs * steps_per_epoch)

    # Decay ends at decay_end_epoch
    decay_end_steps_cfg = sched_cfg.get("decay_end_steps")
    if decay_end_steps_cfg is not None:
        decay_end_steps = int(decay_end_steps_cfg)
    else:
        decay_end_epoch = float(sched_cfg.get("decay_end_epoch", training_cfg.get("decay_end_epoch", warmup_steps / steps_per_epoch if steps_per_epoch else 0)))
        decay_end_steps = int(decay_end_epoch * steps_per_epoch)

    initial_phase_steps = max(initial_phase_steps, 0)
    warmup_steps = max(warmup_steps, initial_phase_steps)  # warmup must be >= initial phase
    decay_end_steps = max(decay_end_steps, warmup_steps)
    total_decay_steps = max(decay_end_steps - warmup_steps, 1)

    for group in optimizer.param_groups:
        group["lr"] = base_lr

    if schedule_type == "linear":

        def lr_lambda(step: int) -> float:
            # Phase 1: Initial phase (constant initial_lr)
            if step < initial_phase_steps:
                return initial_ratio
            # Phase 2: Warmup/constant phase (constant base_lr)
            if step < warmup_steps:
                return 1.0
            # Phase 4: Post-decay (constant final_lr)
            if step >= decay_end_steps:
                return final_ratio
            # Phase 3: Linear decay
            progress = (step - warmup_steps) / total_decay_steps
            return 1.0 - (1.0 - final_ratio) * progress

    elif schedule_type == "cosine":

        def lr_lambda(step: int) -> float:
            # Phase 1: Initial phase (constant initial_lr)
            if step < initial_phase_steps:
                return initial_ratio
            # Phase 2: Warmup/constant phase (constant base_lr)
            if step < warmup_steps:
                return 1.0
            # Phase 4: Post-decay (constant final_lr)
            if step >= decay_end_steps:
                return final_ratio
            # Phase 3: Cosine decay
            progress = (step - warmup_steps) / total_decay_steps
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return final_ratio + (1.0 - final_ratio) * cosine
    else:
        raise ValueError(f"Unsupported scheduler '{schedule_type}'. Choose from ['linear', 'cosine'].")
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    if state_dict is not None:
        scheduler.load_state_dict(state_dict)
    # return some debug msg for optimizer/scheduler
    if initial_phase_steps > 0:
        sched_msg = (f"Scheduler: {schedule_type} with initial_phase_steps={initial_phase_steps} (lr={initial_lr}), "
                    f"warmup_steps={warmup_steps}, decay_end_steps={decay_end_steps}, final_lr={final_lr}")
    else:
        sched_msg = f"Scheduler: {schedule_type} with warmup_steps={warmup_steps}, decay_end_steps={decay_end_steps}, final_lr={final_lr}"
    return scheduler, sched_msg

