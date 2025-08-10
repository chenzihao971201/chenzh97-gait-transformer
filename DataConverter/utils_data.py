"""
utils_data.py
-------------
Data processing utilities for CSI-based gait tasks:
- Reading .mat/.dat files
- Decoding Intel 5300 CSI payloads
- Simple RSS/CSI scaling helpers
- CLI to convert a whole .dat directory to .mat and print a summary
- Parallel conversion support (ProcessPoolExecutor)
- Selectable compressed/uncompressed .mat saving (do_compression)
- read_dat: read .dat into tensors without saving
- read_dat_and_save_mat: convenience wrapper around dat_to_mat_int_batch
"""

from __future__ import annotations
import os
import re
import math
from typing import List, Tuple

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import scipy.io
import torch
from tqdm import tqdm

__all__ = [
    "torch_data",
    "read_mat",
    "read_mat_complex",
    "dat_to_mat_int_batch",
    "gait_datread",
    "read_bf_file",
    "parse_csi",
    "parse_csi_new",
    "db",
    "dbinv",
    "get_total_rss",
    "get_scale_csi",
    "read_dat",                 # ← only read .dat to Tensor (no saving)
    "read_dat_and_save_mat",    # ← wrapper to save .mat (calls dat_to_mat_int_batch)
]


# ================================
# Data I/O and preprocessing
# ================================
def torch_data(X_complex: np.ndarray) -> torch.Tensor:
    """
    Convert a complex array [B, 2000, 30, 3] into a real-imag split
    tensor [B, 2000, 90, 2] where the last dim is [real, imag].
    """
    batch, t, ch, three = X_complex.shape
    assert three == 3, "The last dimension must be 3 (e.g., 3 streams/antennas)."

    X_flat = X_complex.reshape(batch, t, ch * three)
    real = np.real(X_flat)
    imag = np.imag(X_flat)
    return torch.tensor(np.stack([real, imag], axis=-1), dtype=torch.float32)


def read_mat(folder: str, pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Read .mat files named as userA-B-C-rX.mat, extract 'CSI_mat',
    right-pad sequences to the max length, and return stacked tensors.
    """
    files = sorted([f for f in os.listdir(folder) if f.endswith(".mat")])

    arrays, labels = [], []
    for f in tqdm(files, desc="Reading .mat files"):
        m = re.match(r"user(\d+)-\d+-\d+-r\d+\.mat", f)
        if not m:
            print(f"Filename not matched: {f}, skipping.")
            continue

        label = int(m.group(1)) - 1
        arr = scipy.io.loadmat(os.path.join(folder, f))["CSI_mat"]  # [len, 90, 2]
        arrays.append(torch.tensor(arr, dtype=torch.float32))
        labels.append(label)

    if not arrays:
        return torch.empty(0), torch.empty(0, dtype=torch.long)

    max_len = max(a.shape[0] for a in arrays)
    padded = []
    for a in arrays:
        pad_len = max_len - a.shape[0]
        if pad_len > 0:
            pad = torch.full((pad_len, a.shape[1], a.shape[2]), pad_value, dtype=a.dtype)
            a = torch.cat([a, pad], dim=0)
        padded.append(a)

    data_tensor = torch.stack(padded)          # [N, max_len, 90, 2]
    labels_tensor = torch.tensor(labels)       # [N]
    return data_tensor, labels_tensor


def read_mat_complex(folder: str, pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Read complex CSI .mat files, split into real & imaginary parts,
    flatten subcarrier and antenna dims, then pad sequences to the same length.
    """
    # List and sort all .mat files in the folder
    files = sorted(f for f in os.listdir(folder) if f.endswith(".mat"))
    arrays, labels = [], []

    for filename in tqdm(files, desc="Reading .mat files"):
        # Extract user ID from filename, e.g. 'user3-2-1-r1.mat'
        match = re.match(r"user(\d+)-\d+-\d+-r\d+\.mat", filename)
        if not match:
            continue
        label = int(match.group(1)) - 1

        # 1) Load the complex CSI data array of shape (T, 30, 3)
        mat_contents = scipy.io.loadmat(os.path.join(folder, filename))
        csi_complex = mat_contents["CSI_mat"]  # complex ndarray

        # 2) Separate real and imaginary components, shape still (T, 30, 3)
        real_part = np.real(csi_complex)
        imag_part = np.imag(csi_complex)

        # 3) Stack real and imaginary along a new last axis → (T, 30, 3, 2)
        stacked = np.stack([real_part, imag_part], axis=-1)

        # 4) Reshape to merge subcarrier and antenna dims → (T, 90, 2)
        T = stacked.shape[0]
        arr = stacked.reshape(T, -1, 2)  # 30 * 3 = 90 features

        arrays.append(torch.tensor(arr, dtype=torch.float32))
        labels.append(label)

    if not arrays:
        return torch.empty(0), torch.empty(0, dtype=torch.long)

    max_len = max(a.shape[0] for a in arrays)
    padded = []
    for a in arrays:
        pad_len = max_len - a.shape[0]
        if pad_len > 0:
            pad = torch.full((pad_len, a.shape[1], a.shape[2]), pad_value, dtype=a.dtype)
            a = torch.cat([a, pad], dim=0)
        padded.append(a)

    data_tensor = torch.stack(padded)          # [N, max_len, 90, 2]
    labels_tensor = torch.tensor(labels)       # [N]
    return data_tensor, labels_tensor

# ---------------- Parallel helper ----------------
def _convert_one(task: tuple[str, str, bool]) -> dict:
    """
    Worker: read a single .dat, write .mat, return summary info.
    task = (in_path, out_path, do_compression)
    Returns dict with keys: src, dst, shape (and optional error).
    """
    in_path, out_path, do_compression = task
    try:
        arr = read_bf_file(in_path)  # torch.FloatTensor [len, 90, 2]
        np_arr = arr.cpu().numpy() if isinstance(arr, torch.Tensor) else np.asarray(arr)
        scipy.io.savemat(out_path, {"CSI_mat": np_arr}, do_compression=do_compression)
        return {"src": in_path, "dst": out_path, "shape": tuple(np_arr.shape)}
    except Exception as e:
        return {"src": in_path, "dst": out_path, "shape": None, "error": repr(e)}


def dat_to_mat_int_batch(
    input_folder: str = "data",
    output_folder: str = "datamat",
    return_info: bool = False,
    num_workers: int = 1,
    do_compression: bool = False,
):
    """
    Convert all .dat in `input_folder` to .mat in `output_folder` with var 'CSI_mat'.
    Optionally returns per-file summary (source, target, shape).
    Parallelism is controlled by `num_workers` (1 = serial). We clamp workers to ≤ 8.
    Set `do_compression=True` to enable zlib compression in .mat files.
    """
    os.makedirs(output_folder, exist_ok=True)

    def natural_key(s: str):
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

    dat_files = sorted(
        [f for f in os.listdir(input_folder) if f.endswith(".dat")],
        key=natural_key,
    )

    tasks = [
        (os.path.join(input_folder, fname),
         os.path.join(output_folder, os.path.splitext(fname)[0] + ".mat"),
         do_compression)
        for fname in dat_files
    ]

    results = []
    desc = "Converting .dat to .mat"

    # Clamp workers to [1, 8]（避免过多并发写盘）
    if num_workers is None or num_workers <= 1:
        for in_path, out_path, comp in tqdm(tasks, desc=desc, ncols=80):
            results.append(_convert_one((in_path, out_path, comp)))
    else:
        max_workers = min(int(num_workers), 8)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_convert_one, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, ncols=80):
                results.append(fut.result())

        # Sort by original file name for readable output
        results.sort(key=lambda x: natural_key(os.path.basename(x["src"])) if x.get("src") else "")

    if return_info:
        return results


# ---------------- New: read .dat as Tensors (no saving) ----------------
def read_dat(
    folder: str,
    pattern: str | None = None,
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Read .dat files from `folder` (optionally filtered by regex `pattern`),
    decode to tensors, right-pad to max length, and return:
        data_tensor: [N, max_len, 90, 2]
        filenames:   list[str]  (relative names in `folder`)
    This function DOES NOT save .mat to disk.
    """
    all_files = sorted(f for f in os.listdir(folder) if f.endswith(".dat"))
    selected = [f for f in all_files if (pattern is None or re.search(pattern, f))]

    data_list = []
    kept_names = []
    for f in tqdm(selected, desc="Reading .dat (tensor only)"):
        arr = read_bf_file(os.path.join(folder, f))  # [len, 90, 2] (float32)
        if isinstance(arr, torch.Tensor) and arr.numel() == 0:
            print(f"WARN: empty CSI in {f}, skipped.")
            continue
        data_list.append(arr if isinstance(arr, torch.Tensor) else torch.tensor(arr, dtype=torch.float32))
        kept_names.append(f)

    if not data_list:
        return torch.empty(0), []

    max_size = max(d.shape[0] for d in data_list)
    padded_list = []
    for d in data_list:
        pad_len = max_size - d.shape[0]
        if pad_len > 0:
            pad = torch.full((pad_len, d.shape[1], d.shape[2]), pad_value, dtype=d.dtype)
            d = torch.cat([d, pad], dim=0)
        padded_list.append(d)

    data_tensor = torch.stack(padded_list)  # [N, max_len, 90, 2]
    return data_tensor, kept_names


def read_dat_and_save_mat(
    input_folder: str,
    output_folder: str,
    return_info: bool = True,
    num_workers: int = 1,
    do_compression: bool = False,
):
    """
    Compatibility wrapper: read all .dat under `input_folder` and SAVE .mat to `output_folder`.
    This simply forwards to `dat_to_mat_int_batch` with the same behavior/signature.
    """
    return dat_to_mat_int_batch(
        input_folder=input_folder,
        output_folder=output_folder,
        return_info=return_info,
        num_workers=num_workers,
        do_compression=do_compression,
    )


def gait_datread(
    folder: str,
    users: List[int] | None = None,
    num_path: int = 4,
    num_instance: int = 10,
    r_num: int = 1,
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Load files user{user}-{b}-{c}-r{r_num}.dat, decode to tensors, and right-pad.
    NOTE: This function DOES NOT save .mat files; it only returns tensors and labels.
    """
    if users is None:
        users = [1]

    files = set(os.listdir(folder))
    selected, label_list = [], []
    for user in users:
        for b in range(1, num_path + 1):
            for c in range(1, num_instance + 1):
                fname = f"user{user}-{b}-{c}-r{r_num}.dat"
                if fname in files:
                    selected.append(fname)
                    label_list.append(user - 1)

    data_list = []
    for f in tqdm(selected, desc="Reading files"):
        arr = read_bf_file(os.path.join(folder, f))  # [len, 90, 2]
        data_list.append(arr)

    if not data_list:
        return torch.empty(0), torch.empty(0, dtype=torch.long), []

    max_size = max(d.shape[0] for d in data_list)
    padded_list = []
    for d in data_list:
        pad_len = max_size - d.shape[0]
        if pad_len > 0:
            pad = torch.full((pad_len, d.shape[1], d.shape[2]), pad_value, dtype=d.dtype)
            d = torch.cat([d, pad], dim=0)
        padded_list.append(d)

    data_tensor = torch.stack(padded_list)  # [N, max_len, 90, 2]
    labels_tensor = torch.tensor(label_list, dtype=torch.long)
    return data_tensor, labels_tensor, selected


# ================================
# CSI decoding utilities
# ================================
def read_bf_file(filename: str, decoder: str = "python") -> torch.Tensor:
    """
    Read Intel 5300-style .dat (beamforming) records, parse CSI for code==187,
    and return [num_records, 90, 2] where the last dim is [real, imag].
    """
    with open(filename, "rb") as f:
        bfee_list = []
        field_len = int.from_bytes(f.read(2), byteorder="big", signed=False)
        while field_len != 0:
            bfee_list.append(f.read(field_len))
            field_len = int.from_bytes(f.read(2), byteorder="big", signed=False)

    recs = []
    triangle = [0, 1, 3]

    for array in bfee_list:
        code = array[0]
        if code != 187:
            continue

        timestamp_low = int.from_bytes(array[1:5], byteorder="little", signed=False)
        bfee_count = int.from_bytes(array[5:7], byteorder="little", signed=False)
        Nrx, Ntx = array[9], array[10]
        rssi_a, rssi_b, rssi_c = array[11], array[12], array[13]
        noise = array[14] - 256
        agc = array[15]
        antenna_sel = array[16]
        b_len = int.from_bytes(array[17:19], byteorder="little", signed=False)
        fake_rate_n_flags = int.from_bytes(array[19:21], byteorder="little", signed=False)
        payload = array[21:]

        calc_len = (30 * (Nrx * Ntx * 8 * 2 + 3) + 6) / 8
        perm = [0, 0, 0]
        perm[0] = (antenna_sel) & 0x3
        perm[1] = (antenna_sel >> 2) & 0x3
        perm[2] = (antenna_sel >> 4) & 0x3

        if b_len != calc_len:
            print("MIMOToolbox:read_bfee_new:size Wrong beamforming matrix size.")

        if decoder == "python":
            csi = parse_csi(payload, Ntx, Nrx)  # [Ntx, Nrx, 30]
        else:
            print("decoder name error! Wrong decoder:", decoder)
            return torch.empty(0)

        if 1 <= Nrx <= 3 and sum(perm) == triangle[Nrx - 1]:
            csi[:, perm, :] = csi[:, [0, 1, 2], :]
        else:
            print(f"WARN: Found CSI ({filename}) with Nrx={Nrx} and invalid perm={perm}")

        recs.append(
            {
                "timestamp_low": timestamp_low,
                "bfee_count": bfee_count,
                "Nrx": Nrx,
                "Ntx": Ntx,
                "rssi_a": rssi_a,
                "rssi_b": rssi_b,
                "rssi_c": rssi_c,
                "noise": noise,
                "agc": agc,
                "antenna_sel": antenna_sel,
                "perm": perm,
                "len": b_len,
                "fake_rate_n_flags": fake_rate_n_flags,
                "calc_len": calc_len,
                "csi": csi,
            }
        )

    if not recs:
        return torch.empty(0, 90, 2)

    data = np.stack(
        [
            np.stack([rec["csi"].real, rec["csi"].imag], axis=-1).reshape(-1, 2)
            for rec in recs
        ],
        axis=0,
    )
    return torch.tensor(data, dtype=torch.float32)


def parse_csi(payload: bytes, Ntx: int, Nrx: int) -> np.ndarray:
    """
    Parse payload into complex CSI of shape [Ntx, Nrx, 30].
    """
    csi = np.zeros((Ntx, Nrx, 30), dtype=np.complex64)
    index = 0
    for sc in range(30):
        index += 3
        remainder = index % 8
        for j in range(Nrx):
            for k in range(Ntx):
                start = index // 8
                real_bin = bytes([(payload[start] >> remainder) | ((payload[start + 1] << (8 - remainder)) & 0xFF)])
                imag_bin = bytes([(payload[start + 1] >> remainder) | ((payload[start + 2] << (8 - remainder)) & 0xFF)])
                real = int.from_bytes(real_bin, byteorder="little", signed=True)
                imag = int.from_bytes(imag_bin, byteorder="little", signed=True)
                csi[k, j, sc] = np.complex64(complex(float(real), float(imag)))
                index += 16
    return csi


def parse_csi_new(payload: bytes, Ntx: int, Nrx: int) -> np.ndarray:
    """
    Alternative parser that returns shape [30, Nrx, Ntx].
    """
    csi = np.zeros((30, Nrx, Ntx), dtype=np.complex64)
    index = 0
    for sc in range(30):
        index += 3
        remainder = index % 8
        for j in range(Nrx):
            for k in range(Ntx):
                start = index // 8
                real_bin = bytes([(payload[start] >> remainder) | ((payload[start + 1] << (8 - remainder)) & 0xFF)])
                imag_bin = bytes([(payload[start + 1] >> remainder) | ((payload[start + 2] << (8 - remainder)) & 0xFF)])
                real = int.from_bytes(real_bin, byteorder="little", signed=True)
                imag = int.from_bytes(imag_bin, byteorder="little", signed=True)
                csi[sc, j, k] = np.complex64(complex(float(real), float(imag)))
                index += 16
    return csi


# ================================
# RSS / scaling helpers
# ================================
def db(X: float, unit: str) -> float:
    """
    Convert to dB. If unit starts with 'power', X is treated as power (>=0).
    Otherwise, treat X as a magnitude and square it.
    """
    R = 1
    if "power".startswith(unit):
        assert X >= 0
    else:
        X = math.pow(abs(X), 2) / R
    return (10 * math.log10(X) + 300) - 300


def dbinv(x: float) -> float:
    """Inverse of dB conversion for power quantities."""
    return math.pow(10, x / 10)


def get_total_rss(csi_st: dict) -> float:
    """
    Aggregate RSS from rssi_a/b/c (in dB), convert to mW, and return total RSS in dB.
    """
    rssi_mag = 0.0
    if csi_st["rssi_a"] != 0:
        rssi_mag += dbinv(csi_st["rssi_a"])
    if csi_st["rssi_b"] != 0:
        rssi_mag += dbinv(csi_st["rssi_b"])
    if csi_st["rssi_c"] != 0:
        rssi_mag += dbinv(csi_st["rssi_c"])
    return db(rssi_mag, "power") - 44 - csi_st["agc"]


def get_scale_csi(csi_st: dict) -> np.ndarray:
    """
    Scale CSI using RSS to approximate absolute channel magnitudes.
    """
    csi = csi_st["csi"]  # [Ntx, Nrx, 30]

    csi_sq = np.multiply(csi, np.conj(csi)).real
    csi_pwr = np.sum(csi_sq, axis=0).reshape(1, -1, 30)  # [1, Nrx, 30]

    rssi_pwr = dbinv(get_total_rss(csi_st))
    scale = rssi_pwr / (csi_pwr / 30)

    noise_db = -92 if csi_st["noise"] == -127 else csi_st["noise"]
    thermal_noise_pwr = dbinv(noise_db)

    quant_error_pwr = scale * (csi_st["Nrx"] * csi_st["Ntx"])
    total_noise_pwr = thermal_noise_pwr + quant_error_pwr

    ret = csi * np.sqrt(scale / total_noise_pwr)
    if csi_st["Ntx"] == 2:
        ret = ret * math.sqrt(2)
    elif csi_st["Ntx"] == 3:
        ret = ret * math.sqrt(dbinv(4.5))
    return ret


# ================================
# Script entry (CLI via main)
# ================================
if __name__ == "__main__":
    # Example: Set your input/output folders
    input_folder = "./Gait/data/CSI_Gait_dat/user1"
    output_folder = "./Gait/data/CSI_Gait_mat_int/user1"
    os.makedirs(output_folder, exist_ok=True)

    # Choose parallelism: keep 1 core free, but cap at 8
    try:
        cpu_cnt = os.cpu_count() or 1
    except Exception:
        cpu_cnt = 1
    num_workers = min(8, max(1, cpu_cnt - 1))

    # Whether to compress .mat output (True=smaller files, slight CPU cost)
    compress_output = False  # ← set False for uncompressed .mat files

    # Convert and collect per-file info
    info = dat_to_mat_int_batch(
        input_folder=input_folder,
        output_folder=output_folder,
        return_info=True,
        num_workers=num_workers,
        do_compression=compress_output,
    ) or []

    # Print a readable summary
    if not info:
        print(f"No .dat files converted (input='{input_folder}').")
    else:
        print(f"\nConverted {len(info)} files (compression={compress_output}):")
        for i, it in enumerate(info, 1):
            if it.get("error"):
                print(f"[{i:03d}] {os.path.basename(it['src'])} -> ERROR: {it['error']}")
            else:
                print(f"[{i:03d}] {os.path.basename(it['src'])} -> "
                      f"{os.path.basename(it['dst'])}, shape={it['shape']}")
        print("\nDone.")
