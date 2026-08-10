from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import TrainingConfig


def get_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_optimizer(
    model: nn.Module,
    name: str,
    learning_rate: float,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    key = name.lower()
    if key == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if key == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def binary_loss(batch: tuple[torch.Tensor, torch.Tensor], model: nn.Module, device: torch.device) -> torch.Tensor:
    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    logits = model(x)
    return nn.functional.binary_cross_entropy_with_logits(logits, y)


def masked_multioutput_loss(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    x, y, mask = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    logits = model(x)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


def recommender_loss(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    x, antibiotic_idx, y = batch
    x = x.to(device, non_blocking=True)
    antibiotic_idx = antibiotic_idx.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    logits = model(x, antibiotic_idx)
    return nn.functional.binary_cross_entropy_with_logits(logits, y)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        loss = loss_fn(batch, model, device)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def train_with_early_stopping(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    loss_fn: Callable,
    optimizer_name: str,
    learning_rate: float,
    training_config: TrainingConfig,
    device: torch.device,
) -> tuple[nn.Module, dict[str, float]]:
    model = model.to(device)
    optimizer = make_optimizer(
        model,
        name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=training_config.weight_decay,
    )
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
    best_val_loss = float("inf")
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
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= training_config.patience:
            break

    model.load_state_dict(best_state)
    history = {"best_val_loss": best_val_loss, "best_epoch": float(best_epoch)}
    return model, history


@torch.no_grad()
def predict_binary(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    dataset = torch.as_tensor(X, dtype=torch.float32)
    preds: list[np.ndarray] = []
    for start in range(0, dataset.shape[0], batch_size):
        batch = dataset[start : start + batch_size].to(device)
        prob = torch.sigmoid(model(batch)).detach().cpu().numpy()
        preds.append(prob)
    return np.concatenate(preds, axis=0)


@torch.no_grad()
def predict_multioutput(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    dataset = torch.as_tensor(X, dtype=torch.float32)
    preds: list[np.ndarray] = []
    for start in range(0, dataset.shape[0], batch_size):
        batch = dataset[start : start + batch_size].to(device)
        prob = torch.sigmoid(model(batch)).detach().cpu().numpy()
        preds.append(prob)
    return np.concatenate(preds, axis=0)


@torch.no_grad()
def predict_recommender(
    model: nn.Module,
    X: np.ndarray,
    n_antibiotics: int,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    outputs = np.full((X.shape[0], n_antibiotics), np.nan, dtype=np.float32)
    x_tensor = torch.as_tensor(X, dtype=torch.float32)
    for antibiotic_idx in range(n_antibiotics):
        antibiotic_tensor = torch.full((X.shape[0],), antibiotic_idx, dtype=torch.long)
        preds: list[np.ndarray] = []
        for start in range(0, X.shape[0], batch_size):
            x_batch = x_tensor[start : start + batch_size].to(device)
            a_batch = antibiotic_tensor[start : start + batch_size].to(device)
            prob = torch.sigmoid(model(x_batch, a_batch)).detach().cpu().numpy()
            preds.append(prob)
        outputs[:, antibiotic_idx] = np.concatenate(preds, axis=0)
    return outputs

