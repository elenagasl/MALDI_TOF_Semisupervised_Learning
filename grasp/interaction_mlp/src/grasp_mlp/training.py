from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import TrainingConfig
from .metrics import patient_auc


def get_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_optimizer(model: nn.Module, name: str, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    key = name.lower()
    if key == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if key == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if key == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def masked_multioutput_loss(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    x, species_idx, y, mask = batch
    x = x.to(device, non_blocking=True)
    species_idx = species_idx.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    logits = model(x, species_idx)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, loss_fn: Callable, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        loss = loss_fn(batch, model, device)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


@torch.no_grad()
def predict_species_aware_mlp(
    model: nn.Module,
    X: np.ndarray,
    species_codes: np.ndarray,
    sample_indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    x_tensor = torch.as_tensor(X[sample_indices], dtype=torch.float32)
    species_tensor = torch.as_tensor(species_codes[sample_indices], dtype=torch.long)
    preds: list[np.ndarray] = []
    for start in range(0, len(sample_indices), batch_size):
        x_batch = x_tensor[start : start + batch_size].to(device, non_blocking=True)
        s_batch = species_tensor[start : start + batch_size].to(device, non_blocking=True)
        prob = torch.sigmoid(model(x_batch, s_batch)).detach().cpu().numpy()
        preds.append(prob.astype(np.float32))
    return np.concatenate(preds, axis=0) if preds else np.empty((0, 0), dtype=np.float32)


def evaluate_patient_auc(
    model: nn.Module,
    X: np.ndarray,
    species_codes: np.ndarray,
    amr: np.ndarray,
    sample_indices: np.ndarray,
    antibiotic_indices: list[int],
    device: torch.device,
    batch_size: int,
) -> float:
    if len(sample_indices) == 0 or not antibiotic_indices:
        return float("nan")
    y_true = amr[np.ix_(sample_indices, antibiotic_indices)]
    y_score = predict_species_aware_mlp(model, X, species_codes, sample_indices, device, batch_size)
    return patient_auc(y_true, y_score)


def train_with_early_stopping(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    loss_fn: Callable,
    optimizer_name: str,
    learning_rate: float,
    training_config: TrainingConfig,
    device: torch.device,
    val_X: np.ndarray,
    val_species_codes: np.ndarray,
    val_amr: np.ndarray,
    val_sample_indices: np.ndarray,
    val_antibiotic_indices: list[int],
) -> tuple[nn.Module, dict[str, object]]:
    model = model.to(device)
    optimizer = make_optimizer(model, optimizer_name, learning_rate, training_config.weight_decay)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory and device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory and device.type == "cuda",
    )

    best_state = copy.deepcopy(model.state_dict())
    best_score = -float("inf")
    best_val_loss = float("inf")
    best_val_patient_auc = float("nan")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, training_config.max_epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(batch, model, device)
            loss.backward()
            optimizer.step()

        val_loss = evaluate_loss(model, val_loader, loss_fn, device)
        val_patient_auc = float("nan")
        if training_config.early_stopping_metric == "patient_auc":
            val_patient_auc = evaluate_patient_auc(
                model,
                val_X,
                val_species_codes,
                val_amr,
                val_sample_indices,
                val_antibiotic_indices,
                device,
                training_config.prediction_batch_size,
            )
            score = -val_loss if np.isnan(val_patient_auc) else val_patient_auc
        elif training_config.early_stopping_metric == "loss":
            score = -val_loss
        else:
            raise ValueError(f"Unsupported early_stopping_metric: {training_config.early_stopping_metric}")

        if score > best_score:
            best_score = score
            best_val_loss = val_loss
            best_val_patient_auc = val_patient_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= training_config.patience:
            break

    model.load_state_dict(best_state)
    return model, {
        "best_val_loss": best_val_loss,
        "best_val_patient_auc": best_val_patient_auc,
        "best_epoch": float(best_epoch),
        "early_stopping_metric": training_config.early_stopping_metric,
    }


def finetune_training_config(cfg: TrainingConfig) -> TrainingConfig:
    return TrainingConfig(
        batch_size=cfg.batch_size,
        prediction_batch_size=cfg.prediction_batch_size,
        max_epochs=cfg.finetune_epochs,
        patience=cfg.finetune_patience,
        finetune_epochs=cfg.finetune_epochs,
        finetune_patience=cfg.finetune_patience,
        weight_decay=cfg.weight_decay,
        early_stopping_metric=cfg.early_stopping_metric,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

