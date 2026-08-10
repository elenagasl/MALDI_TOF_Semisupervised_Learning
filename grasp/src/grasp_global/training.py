from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import TrainingConfig
from .metrics import safe_auc


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
    if name.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def interaction_loss(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    x, species_idx, antibiotic_idx, y = batch
    x = x.to(device, non_blocking=True)
    species_idx = species_idx.to(device, non_blocking=True)
    antibiotic_idx = antibiotic_idx.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    logits = model(x, species_idx, antibiotic_idx)
    return nn.functional.binary_cross_entropy_with_logits(logits, y)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, loss_fn: Callable, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        loss = loss_fn(batch, model, device)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def patient_auc_from_matrix(y_true: np.ndarray, y_score: np.ndarray) -> float:
    values: list[float] = []
    for row in range(y_true.shape[0]):
        auc = safe_auc(y_true[row], y_score[row])
        if not np.isnan(auc):
            values.append(auc)
    return float(np.mean(values)) if values else float("nan")


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
    if len(sample_indices) == 0 or len(antibiotic_indices) == 0:
        return float("nan")
    y_true = amr[np.ix_(sample_indices, antibiotic_indices)]
    y_score = predict_grasp(
        model=model,
        X=X,
        species_codes=species_codes,
        sample_indices=sample_indices,
        n_antibiotics=len(antibiotic_indices),
        device=device,
        batch_size=batch_size,
    )
    return patient_auc_from_matrix(y_true, y_score)


def train_with_early_stopping(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    loss_fn: Callable,
    optimizer_name: str,
    learning_rate: float,
    training_config: TrainingConfig,
    device: torch.device,
    val_X: np.ndarray | None = None,
    val_species_codes: np.ndarray | None = None,
    val_amr: np.ndarray | None = None,
    val_sample_indices: np.ndarray | None = None,
    val_antibiotic_indices: list[int] | None = None,
) -> tuple[nn.Module, dict[str, object]]:
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
    best_val_patient_auc = float("nan")
    best_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    metric_name = training_config.early_stopping_metric

    for epoch in range(1, training_config.max_epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(batch, model, device)
            loss.backward()
            optimizer.step()

        val_loss = evaluate_loss(model, val_loader, loss_fn, device)
        val_patient_auc = float("nan")
        if metric_name == "patient_auc":
            if (
                val_X is not None
                and val_species_codes is not None
                and val_amr is not None
                and val_sample_indices is not None
                and val_antibiotic_indices is not None
            ):
                val_patient_auc = evaluate_patient_auc(
                    model=model,
                    X=val_X,
                    species_codes=val_species_codes,
                    amr=val_amr,
                    sample_indices=val_sample_indices,
                    antibiotic_indices=val_antibiotic_indices,
                    device=device,
                    batch_size=training_config.prediction_batch_size,
                )
            if np.isnan(val_patient_auc):
                score = -val_loss
            else:
                score = val_patient_auc
        elif metric_name == "loss":
            score = -val_loss
        else:
            raise ValueError(f"Unsupported early_stopping_metric: {metric_name}")

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
        "early_stopping_metric": metric_name,
    }


@torch.no_grad()
def predict_grasp(
    model: nn.Module,
    X: np.ndarray,
    species_codes: np.ndarray,
    sample_indices: np.ndarray,
    n_antibiotics: int,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    outputs = np.full((len(sample_indices), n_antibiotics), np.nan, dtype=np.float32)
    x_tensor = torch.as_tensor(X[sample_indices], dtype=torch.float32)
    species_tensor = torch.as_tensor(species_codes[sample_indices], dtype=torch.long)

    for antibiotic_idx in range(n_antibiotics):
        antibiotic_tensor = torch.full((len(sample_indices),), antibiotic_idx, dtype=torch.long)
        preds: list[np.ndarray] = []
        for start in range(0, len(sample_indices), batch_size):
            x_batch = x_tensor[start : start + batch_size].to(device)
            s_batch = species_tensor[start : start + batch_size].to(device)
            a_batch = antibiotic_tensor[start : start + batch_size].to(device)
            prob = torch.sigmoid(model(x_batch, s_batch, a_batch)).detach().cpu().numpy()
            preds.append(prob)
        outputs[:, antibiotic_idx] = np.concatenate(preds, axis=0)
    return outputs
