# test.py
# Evaluate TransformerE using the SAME data loading logic as training.

import os
import re
import warnings
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from DataConverter.utils_data import read_mat_complex  # same as training
from Model.models import TransformerE                  # your model class

# ---------------- Suppress non-critical warnings ----------------
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.nn\.modules\.transformer")
warnings.filterwarnings("ignore", message=r"Attempting to run cuBLAS.*")

# -------- Fixed config (match training) --------
DATA_ROOT   = ".data/CSI_Gait_mat_double"
USER_GLOB   = "user*"
USER_START  = 1
USER_END    = 11
SEQ_LEN     = 2000
BATCH_SIZE  = 32
CKPT_PATH   = ".runs/TransformerE/100.pth"

# -------- Helpers (same as training) --------
def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

def list_user_dirs(root: str):
    dirs = [d for d in glob(os.path.join(root, USER_GLOB)) if os.path.isdir(d)]
    return sorted(dirs, key=lambda p: natural_key(os.path.basename(p)))

def pad_or_crop_batch(X: torch.Tensor, target_len: int) -> torch.Tensor:
    # X: [N, T, 90, 2] -> [N, target_len, 90, 2]
    N, T, F, C = X.shape
    if T == target_len:
        return X
    out = torch.zeros((N, target_len, F, C), dtype=X.dtype)
    if T < target_len:
        out[:, :T] = X
        return out
    start = (T - target_len) // 2
    out[:] = X[:, start:start + target_len]
    return out

def load_dataset(root: str, seq_len: int):
    """
    Load .mat sequences for users in [USER_START, USER_END],
    adjust labels, pad/crop to seq_len, and aggregate tensors.
    """
    dirs = list_user_dirs(root)
    filtered = []
    for d in dirs:
        m = re.search(r"user(\d+)", os.path.basename(d))
        if not m:
            continue
        uid = int(m.group(1))
        if USER_START <= uid <= USER_END:
            filtered.append((d, uid))
    if not filtered:
        raise FileNotFoundError(f"No user directories in range {USER_START}-{USER_END}")

    X_list, y_list = [], []
    print(f"Reading users in range [{USER_START}, {USER_END}] from: {root}")
    for d, uid in filtered:
        X_sub, y_sub = read_mat_complex(d)    # [Ni, Ti, 90, 2], [Ni]
        if X_sub.numel() == 0:
            print(f"  user{uid}: empty folder, skipped")
            continue
        orig_len = X_sub.shape[1]
        y_sub = y_sub - (USER_START - 1)      # shift labels so USER_START -> 0
        X_sub = pad_or_crop_batch(X_sub, seq_len)
        print(f"  user{uid}: X={tuple(X_sub.shape)} (orig_len={orig_len} -> target_len={seq_len}), y={tuple(y_sub.shape)}")
        X_list.append(X_sub)
        y_list.append(y_sub)

    if not X_list:
        raise RuntimeError("No valid samples found.")

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    num_classes = USER_END - USER_START + 1
    print(f"Aggregated: X={tuple(X.shape)}, y={tuple(y.shape)}, classes={num_classes}")
    return X, y, num_classes

# -------- Quiet evaluation (avoid DataParallel/key prefix issues) --------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    crit = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = crit(logits, yb)
        loss_sum += loss.item() * xb.size(0)
        correct  += (logits.argmax(1) == yb).sum().item()
        total    += xb.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)

def main():
    # 1) Load data same as training
    X, y, num_classes = load_dataset(DATA_ROOT, SEQ_LEN)

    # 2) Reproduce the same split (same ratios & seed)
    N = len(y)
    n_test = int(N)
    print(f"Split -> test:{n_test}")

    dataset = TensorDataset(X, y)

    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 3) Build model and load checkpoint (single device, strip 'module.' if present)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerE(num_classes=num_classes).to(device)
    state = torch.load(CKPT_PATH, map_location=device)
    if any(k.startswith('module.') for k in state.keys()):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)

    # 4) Evaluate on the test split
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test: loss={test_loss:.4f}, acc={test_acc:.4f}")

if __name__ == "__main__":
    main()
