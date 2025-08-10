# benchmark_io_20_pair.py
# Compare read times on the SAME filenames (up to 20) across:
#   - DAT  (decoded to Tensor via utils_data.read_dat, per-file, no saving)
#   - MAT  (uncompressed, scipy.io.loadmat -> Tensor)
#   - MATc (compressed, scipy.io.loadmat -> Tensor)
# Shows ONLY THREE progress bars (one per category).
# Edit the directories in __main__ before running.

import os
import re
import glob
import time
import random
from contextlib import contextmanager, redirect_stderr
from typing import List, Set, Tuple

import numpy as np
import scipy.io
import torch
from tqdm import tqdm

import utils_data
from utils_data import read_dat  # read-only .dat -> Tensor (no saving)

# ------------------------------- helpers -------------------------------

def basenames_with_ext(root: str, ext: str) -> Set[str]:
    """Return the set of basenames (without extension) for files in `root` with extension `ext`."""
    paths = glob.glob(os.path.join(root, f"*.{ext}")) + glob.glob(os.path.join(root, f"*.{ext.upper()}"))
    return {os.path.splitext(os.path.basename(p))[0] for p in paths}

def sample_basenames(basenames: List[str], k: int, seed: int) -> List[str]:
    """Sample up to k basenames with a fixed RNG seed; return in sorted order for determinism."""
    rng = random.Random(seed)
    if len(basenames) <= k:
        return sorted(basenames)
    return sorted(rng.sample(basenames, k))

def to_tensor_cpu(arr, dtype: torch.dtype) -> torch.Tensor:
    """Convert numpy array to a CPU Tensor."""
    return torch.as_tensor(arr, dtype=dtype, device="cpu")

@contextmanager
def suppress_inner_tqdm():
    """
    Suppress progress bars created inside utils_data.read_dat so only the
    three top-level bars from this script are shown.
    """
    # Monkey-patch the tqdm symbol imported inside utils_data
    orig = utils_data.tqdm
    def _no_tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []
    try:
        utils_data.tqdm = _no_tqdm
        # tqdm writes to stderr by default; redirect to /dev/null just in case
        with open(os.devnull, "w") as devnull, redirect_stderr(devnull):
            yield
    finally:
        utils_data.tqdm = orig

# ------------------------------- timing -------------------------------

def time_mat_paths(paths: List[str], passes: int, dtype: torch.dtype, desc: str) -> Tuple[float, float]:
    """Total time per pass for given .mat paths (mean/std across passes). Single progress bar."""
    totals = []
    steps = passes * len(paths)
    with tqdm(total=steps, desc=desc, ncols=90) as pbar:
        for _ in range(passes):
            order = paths[:]
            random.shuffle(order)
            total = 0.0
            for p in order:
                t0 = time.perf_counter()
                arr = scipy.io.loadmat(p)["csi_data"]
                _ = to_tensor_cpu(arr, dtype)
                total += (time.perf_counter() - t0)
                pbar.update(1)
            totals.append(total)
    return float(np.mean(totals)), float(np.std(totals))

def time_dat_paths(paths: List[str], passes: int, dtype: torch.dtype, desc: str) -> Tuple[float, float]:
    """Total time per pass for given .dat paths (mean/std across passes). Single progress bar."""
    totals = []
    steps = passes * len(paths)
    with tqdm(total=steps, desc=desc, ncols=90) as pbar:
        for _ in range(passes):
            order = paths[:]
            random.shuffle(order)
            total = 0.0
            for dat_path in order:
                folder = os.path.dirname(dat_path)
                basename = os.path.basename(dat_path)
                pattern = r"^" + re.escape(basename) + r"$"
                t0 = time.perf_counter()
                with suppress_inner_tqdm():  # keep only the three top-level bars
                    X, _ = read_dat(folder, pattern=pattern)  # X: [1, T, 90, 2]
                if X.dtype is not dtype:
                    _ = X.to(dtype)
                total += (time.perf_counter() - t0)
                pbar.update(1)
            totals.append(total)
    return float(np.mean(totals)), float(np.std(totals))

# ------------------------------- main -------------------------------

if __name__ == "__main__":
    # === EDIT THESE DIRECTORIES ===
    dat_dir = "./Gait/data/CSI_Gait/user1"
    mat_uncompressed_dir = "./Gait/data/CSI_Gait_mat/user1"       # uncompressed .mat
    mat_compressed_dir   = "./Gait/data/CSI_Gait_mat/user1_cmprsd" # compressed .mat

    K = 20               # number of paired files to include (max)
    PASSES = 3           # full rounds for stability (3–5 recommended)
    DTYPE = torch.float32
    SEED = 42

    # Build common basenames (guaranteed identical names except extension/compression)
    b_dat = basenames_with_ext(dat_dir, "dat")
    b_mu  = basenames_with_ext(mat_uncompressed_dir, "mat")
    b_mc  = basenames_with_ext(mat_compressed_dir, "mat")

    common = sorted(b_dat & b_mu & b_mc)
    if not common:
        raise RuntimeError("No common basenames across DAT / MAT uncompressed / MAT compressed.")
    picked = sample_basenames(common, K, SEED)

    # Construct path lists in the same order
    dat_paths = [os.path.join(dat_dir, f"{b}.dat") for b in picked]
    mu_paths  = [os.path.join(mat_uncompressed_dir, f"{b}.mat") for b in picked]
    mc_paths  = [os.path.join(mat_compressed_dir,   f"{b}.mat") for b in picked]

    print(f"Paired basenames: {len(picked)} (selected from intersection of the three directories)")

    # DAT
    mean_d, std_d = time_dat_paths(dat_paths, passes=PASSES, dtype=DTYPE, desc="DAT (paired subset)")
    print(f"[DAT paired]           files={len(dat_paths):3d}  total mean/std (s): {mean_d:.6f} / {std_d:.6f}")

    # MAT uncompressed
    mean_mu, std_mu = time_mat_paths(mu_paths, passes=PASSES, dtype=DTYPE, desc="MAT uncompressed (paired subset)")
    print(f"[MAT uncompressed]     files={len(mu_paths):3d}  total mean/std (s): {mean_mu:.6f} / {std_mu:.6f}")

    # MAT compressed
    mean_mc, std_mc = time_mat_paths(mc_paths, passes=PASSES, dtype=DTYPE, desc="MAT compressed (paired subset)")
    print(f"[MAT compressed]       files={len(mc_paths):3d}  total mean/std (s): {mean_mc:.6f} / {std_mc:.6f}")

    # Simple ratio
    if mean_mu > 0:
        print(f"compressed / uncompressed ratio: {mean_mc/mean_mu:.3f}")
