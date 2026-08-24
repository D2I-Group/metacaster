
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    lradj: str,
    lr: float,
    total_epochs: int,
) -> None:
    if lradj == "constant":
        lr_adjust = {epoch: lr}
    elif lradj == "manual_schedule":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif lradj == "exponential":
        lr_adjust = {epoch: lr if epoch < 3 else lr * (0.9 ** ((epoch - 3) // 1))}
    elif lradj == "cosine_annealing":
        lr_adjust = {epoch: lr / 2 * (1 + math.cos(epoch / total_epochs * math.pi))}
    else:
        return

    if epoch in lr_adjust:
        new_lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr


@dataclass
class EarlyStopping:

    patience: int = 3
    delta: float = 0.0

    def __post_init__(self) -> None:
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, val_loss: float) -> bool:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            return True
        if score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        self.best_score = score
        self.counter = 0
        return True


class CheckpointManager:

    def __init__(self, strategy: str, save_k: int, path: str) -> None:
        self.strategy = strategy if strategy in {"best", "topk", "all"} else "best"
        self.save_k = save_k
        self.path = path
        self.top_checkpoints: list[tuple[float, str]] = []
        os.makedirs(self.path, exist_ok=True)

    def save(
        self, model: torch.nn.Module, epoch: int, val_loss: float, is_best: bool
    ) -> str | None:
        if self.strategy == "all":
            filename = f"epoch_{epoch}.pth"
            return self._save(model, filename)

        if self.strategy == "topk":
            filename = f"val_{val_loss:.6f}_epoch_{epoch}.pth"
            saved_path = self._save(model, filename)
            self.top_checkpoints.append((val_loss, saved_path))
            self.top_checkpoints.sort(key=lambda x: x[0])
            while len(self.top_checkpoints) > self.save_k:
                _, to_remove = self.top_checkpoints.pop()
                if os.path.exists(to_remove):
                    os.remove(to_remove)
            return saved_path

        if self.strategy == "best" and is_best:
            return self._save(model, "best_checkpoint.pth")
        return None

    def _save(self, model: torch.nn.Module, filename: str) -> str:
        save_path = os.path.join(self.path, filename)
        torch.save(model.state_dict(), save_path)
        return save_path
