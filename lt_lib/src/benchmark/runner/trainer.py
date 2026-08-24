
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from benchmark.registry.losses import get_loss
from benchmark.utils.training import (
    CheckpointManager,
    EarlyStopping,
    adjust_learning_rate,
)


@dataclass
class TrainResult:

    best_model_path: str
    best_val_loss: float
    train_time_sec: float


def _make_decoder_input(
    batch_y: torch.Tensor, label_len: int, pred_len: int, device: torch.device
) -> torch.Tensor:
    dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
    return dec_inp


def _call_model(model: nn.Module, batch_x, batch_x_mark, dec_inp, batch_y_mark):
    try:
        return model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    except TypeError:
        return model(batch_x)


def train(
    model: nn.Module,
    train_loader,
    vali_loader,
    device: torch.device,
    epochs: int,
    patience: int,
    loss_name: str,
    loss_params: dict,
    optimizer: torch.optim.Optimizer,
    lradj: str,
    base_lr: float,
    total_epochs: int,
    label_len: int,
    pred_len: int,
    features: str,
    use_amp: bool,
    checkpoint_dir: str,
    checkpoint_cfg,
) -> TrainResult:
    model.train()
    criterion = get_loss(loss_name, **loss_params)
    early_stopping = EarlyStopping(patience=patience)
    checkpoint_manager = CheckpointManager(
        strategy=checkpoint_cfg.strategy,
        save_k=checkpoint_cfg.save_k,
        path=checkpoint_dir,
    )
    use_amp = use_amp and device.type != "cpu"
    scaler = torch.amp.GradScaler() if use_amp else None

    start_time = time.perf_counter()
    best_val_loss = float("inf")
    for epoch in range(epochs):
        epoch_losses = []
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            if batch_x_mark is not None:
                batch_x_mark = batch_x_mark.float().to(device)
            if batch_y_mark is not None:
                batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = _make_decoder_input(batch_y, label_len, pred_len, device)

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast(device_type=device.type):
                    outputs = _call_model(
                        model, batch_x, batch_x_mark, dec_inp, batch_y_mark
                    )
                    outputs, batch_y_sliced = _slice_pred_target(
                        outputs, batch_y, pred_len, features
                    )
                    loss = criterion(outputs, batch_y_sliced)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = _call_model(
                    model, batch_x, batch_x_mark, dec_inp, batch_y_mark
                )
                outputs, batch_y_sliced = _slice_pred_target(
                    outputs, batch_y, pred_len, features
                )
                loss = criterion(outputs, batch_y_sliced)
                loss.backward()
                optimizer.step()

            epoch_losses.append(loss.item())

        vali_loss = validate(
            model, vali_loader, device, criterion, label_len, pred_len, features
        )
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        current_lr = optimizer.param_groups[0].get("lr", base_lr)
        print(
            f"Epoch {epoch + 1}/{epochs} | train_loss: {train_loss:.6f} | "
            f"val_loss: {vali_loss:.6f} | lr: {current_lr:.6g}"
        )
        is_best = early_stopping.step(vali_loss)
        if is_best:
            best_val_loss = vali_loss
            best_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
            torch.save(model.state_dict(), best_path)
        checkpoint_manager.save(model, epoch + 1, vali_loss, is_best)
        if early_stopping.early_stop:
            break
        adjust_learning_rate(optimizer, epoch + 1, lradj, base_lr, total_epochs)

    train_time = time.perf_counter() - start_time
    best_model_path = f"{checkpoint_dir}/best_checkpoint.pth"
    model.load_state_dict(torch.load(best_model_path))
    return TrainResult(
        best_model_path=best_model_path,
        best_val_loss=best_val_loss,
        train_time_sec=train_time,
    )


def validate(
    model: nn.Module,
    data_loader,
    device: torch.device,
    criterion: nn.Module,
    label_len: int,
    pred_len: int,
    features: str,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            if batch_x_mark is not None:
                batch_x_mark = batch_x_mark.float().to(device)
            if batch_y_mark is not None:
                batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = _make_decoder_input(batch_y, label_len, pred_len, device)
            outputs = _call_model(model, batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs, batch_y_sliced = _slice_pred_target(
                outputs, batch_y, pred_len, features
            )
            loss = criterion(outputs, batch_y_sliced)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def _slice_pred_target(
    outputs: torch.Tensor, batch_y: torch.Tensor, pred_len: int, features: str
):
    f_dim = -1 if features == "MS" else 0
    outputs = outputs[:, -pred_len:, f_dim:]
    batch_y = batch_y[:, -pred_len:, f_dim:]
    return outputs, batch_y
