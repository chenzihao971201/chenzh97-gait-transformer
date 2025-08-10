"""
utils_model.py
--------------
Model utilities for training, evaluation, and checkpoint management.
"""

from __future__ import annotations
import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

__all__ = ["save_model", "train_model", "test_model"]


def save_model(model: nn.Module, model_dir: str, model_name: str, max_keep: int = 1) -> str:
    """
    Save model state_dict and keep only the most recent `max_keep` files.

    Returns:
        The full path of the saved checkpoint.
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)

    # If model is DataParallel, save the underlying module's state_dict
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state, model_path)

    saved = sorted(
        [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))],
        key=lambda x: os.path.getctime(os.path.join(model_dir, x)),
    )
    while len(saved) > max_keep:
        oldest = saved.pop(0)
        try:
            os.remove(os.path.join(model_dir, oldest))
        except OSError:
            pass
    return model_path


def _load_state_dict_compat(model: nn.Module, state_dict: dict) -> None:
    """
    Load state_dict into model. If keys are prefixed with 'module.' (from DataParallel),
    strip the prefix automatically.
    """
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Strip 'module.' prefix if present
        new_state = {}
        for k, v in state_dict.items():
            new_key = k[7:] if k.startswith("module.") else k
            new_state[new_key] = v
        model.load_state_dict(new_state)


def train_model(
    model_dir: str,
    model: nn.Module,
    loader_train: DataLoader,
    epoch: int = 10,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    lr: float = 1e-3,
    max_keep: int = 1,
) -> None:
    """
    Simple training loop for classification with CrossEntropyLoss.
    Saves a checkpoint whenever the current batch loss improves.
    """
    best_loss = float("inf")

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epoch):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        train_bar = tqdm(loader_train, desc=f"Epoch {ep + 1}/{epoch} [Train]", leave=False)
        for xb, yb in train_bar:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

            train_bar.set_postfix(loss=float(loss.item()))

            # Save when batch loss improves
            if float(loss.item()) < best_loss:
                best_loss = float(loss.item())
                acc = (preds == yb).float().mean().item()
                save_model(model, model_dir, f"{acc * 100:.0f}.pth", max_keep=max_keep)

        avg_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        print(f"Epoch {ep + 1}: Train loss = {avg_loss:.4f}, accuracy = {train_acc:.3f}")


def test_model(
    model: nn.Module,
    model_path: str,
    loader_test: DataLoader,
    device: Optional[torch.device] = None,
) -> None:
    """
    Load a state_dict from `model_path` and evaluate accuracy on `loader_test`.
    """
    device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)
    state_dict = torch.load(model_path, map_location=device)
    _load_state_dict_compat(model, state_dict)
    model.eval()

    correct, total = 0, 0
    test_bar = tqdm(loader_test, desc="[Test]", leave=False)
    with torch.no_grad():
        for xb, yb in test_bar:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

    acc = correct / max(total, 1)
    print(f"Test accuracy = {acc:.3f}")
