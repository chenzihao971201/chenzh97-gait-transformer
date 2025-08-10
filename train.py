# train.py
# Train TransformerE on CSI .mat data using provided utilities.
# Supports selecting a user ID range (USER_START to USER_END) and quiet validation/testing.
# While reading data, it prints the user id and tensor shapes per folder.
# Now adds optional parallel loading across user folders.

import os
import re
import warnings
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from DataConverter.utils_data import read_mat_complex  # use complex reader (T,30,3 complex -> [T,90,2])
from Model.models import TransformerE                  # direct import
from Model.utils_model import train_model              # reuse training loop only

# ---------------- Suppress non-critical warnings ----------------
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.nn\.modules\.transformer")
warnings.filterwarnings("ignore", message=r"Attempting to run cuBLAS.*")

# ---------------- Fixed config (no CLI) ----------------
DATA_ROOT         = "./data/CSI_Gait_mat_double"
USER_GLOB         = "user*"         # subfolders like user1 ... user12
USER_START        = 1                 # start user ID inclusive
USER_END          = 11                # end user ID inclusive
OUT_DIR           = "./runs/TransformerE"
SEQ_LEN           = 2000              # unified sequence length for all samples
BATCH_SIZE        = 32
EPOCHS            = 15
NUM_WORKERS       = 8                 # DataLoader workers
TEST_RATIO        = 0.20
LR                = 1e-3              # learning rate for training
USER_READ_WORKERS = 8                 # parallel workers to read user folders (1 = serial)

# ---------------- Helpers ----------------
def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def list_user_dirs(root: str):
    # List and sort user directories
    dirs = [d for d in glob(os.path.join(root, USER_GLOB)) if os.path.isdir(d)]
    return sorted(dirs, key=lambda p: natural_key(os.path.basename(p)))


def pad_or_crop_batch(X: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Right-pad or center-crop each sequence to target_len.
    X: [N, T, 90, 2] -> [N, target_len, 90, 2]
    """
    N, T, F, C = X.shape
    if T == target_len:
        return X
    out = torch.zeros((N, target_len, F, C), dtype=X.dtype)
    if T < target_len:
        out[:, :T] = X  # pad right
        return out
    # center-crop when longer
    start = (T - target_len) // 2
    out[:] = X[:, start:start + target_len]
    return out


def find_latest_checkpoint(model_dir: str) -> str:
    files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    if not files:
        raise FileNotFoundError(f"No .pth checkpoints in {model_dir}")
    files = sorted(files, key=lambda f: os.path.getctime(os.path.join(model_dir, f)))
    return os.path.join(model_dir, files[-1])

# ---------------- Parallel user-folder reader ----------------
def _read_one_user(task):
    """Worker to read a single user folder and return numpy arrays to reduce IPC cost."""
    d, uid, seq_len, user_start = task
    try:
        X_sub, y_sub = read_mat_complex(d)  # tensors: [Ni, Ti, 90, 2], [Ni]
        if X_sub.numel() == 0:
            return None
        orig_len = int(X_sub.shape[1])
        y_sub = y_sub - (user_start - 1)
        X_sub = pad_or_crop_batch(X_sub, seq_len)
        # Convert to numpy for lighter inter-process transfer
        return {
            "uid": uid,
            "X": X_sub.cpu().numpy(),
            "y": y_sub.cpu().numpy(),
            "orig_len": orig_len,
            "dir": d,
        }
    except Exception as e:
        return {"uid": uid, "error": repr(e), "dir": d}

# ---------------- Data loading ----------------
def load_dataset(root: str, seq_len: int, user_workers: int = USER_READ_WORKERS):
    """
    Load .mat sequences for users in [USER_START, USER_END], adjust labels,
    pad/crop to seq_len, and aggregate into tensors. Prints user id and shapes.
    Optionally parallelize across user folders with ProcessPoolExecutor.
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

    print(f"Reading users in range [{USER_START}, {USER_END}] from: {root}")

    tasks = [(d, uid, seq_len, USER_START) for d, uid in filtered]
    results = []

    if user_workers is None or user_workers <= 1:
        # Serial
        for t in tasks:
            res = _read_one_user(t)
            if res is not None:
                results.append(res)
    else:
        maxw = max(1, int(user_workers))
        with ProcessPoolExecutor(max_workers=maxw) as ex:
            futs = [ex.submit(_read_one_user, t) for t in tasks]
            for fut in as_completed(futs):
                res = fut.result()
                if res is not None:
                    results.append(res)

    # Filter errors and sort by uid for stable prints
    ok, errs = [], []
    for r in results:
        if "error" in r:
            errs.append(r)
        else:
            ok.append(r)
    ok.sort(key=lambda r: r["uid"]) 

    # Print per-user info and build tensors
    X_list, y_list = [], []
    for r in ok:
        uid = r["uid"]
        X_sub = torch.from_numpy(r["X"])  # [Ni, L, 90, 2]
        y_sub = torch.from_numpy(r["y"]).long()
        print(f"  user{uid}: X={tuple(X_sub.shape)} (orig_len={r['orig_len']} -> target_len={seq_len}), y={tuple(y_sub.shape)}")
        X_list.append(X_sub)
        y_list.append(y_sub)

    # Report any errors
    for r in errs:
        print(f"  user{r['uid']}: ERROR reading {r['dir']}: {r['error']}")

    if not X_list:
        raise RuntimeError("No valid samples found.")

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    num_classes = USER_END - USER_START + 1
    print(f"Aggregated: X={tuple(X.shape)}, y={tuple(y.shape)}, classes={num_classes}")
    return X, y

# ---------------- Evaluation (quiet) ----------------
@torch.no_grad()
def evaluate_quiet(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss_sum += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)

# ---------------- Main ----------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load and split data (optionally parallel across users)
    X, y = load_dataset(DATA_ROOT, SEQ_LEN, user_workers=USER_READ_WORKERS)
    N = len(y)
    n_test = int(N * TEST_RATIO)
    n_train = N - n_test
    print(f"Split -> train:{n_train}  test:{n_test}")

    dataset = TensorDataset(X, y)
    train_set, test_set = random_split(
        dataset, [n_train, n_test], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Build model
    num_classes = USER_END - USER_START + 1
    model = TransformerE(num_classes=num_classes)

    # Train model (utils_model handles DataParallel internally)
    print("\n==> Training TransformerE ...")
    train_model(
        model_dir=OUT_DIR,
        model=model,
        loader_train=train_loader,
        epoch=EPOCHS,
        lr=LR,
        max_keep=1
    )

    # Test quietly
    try:
        ckpt = find_latest_checkpoint(OUT_DIR)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tm = TransformerE(num_classes=num_classes).to(device)
        state = torch.load(ckpt, map_location=device)
        if any(k.startswith('module.') for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        tm.load_state_dict(state, strict=True)
        t_loss, t_acc = evaluate_quiet(tm, test_loader, device)
        print(f"Test: loss={t_loss:.4f}, acc={t_acc:.4f}")
    except Exception:
        print("[Warn] Test skipped")

    print("\nDone.")

if __name__ == "__main__":
    main()