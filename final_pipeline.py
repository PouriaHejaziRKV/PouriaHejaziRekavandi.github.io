# -*- coding: utf-8 -*-
"""
FINAL COLAB PIPELINE
====================
Methods:
1) Tikhonov baseline
2) dSPM
3) ESINet ConvDip (explicitly verified; no silent default-model fallback)
4) Proposed physics-guided, time-resolved residual cortical graph network

Synthetic data have source ground truth. Real EEG does not; real-data dSPM
metrics quantify agreement with a classical reference, not true accuracy.
"""

try:
    import IPython
    IPython.get_ipython().system('pip install "mne>=1.6" "esinet==0.3.0" pandas seaborn matplotlib scikit-learn scipy tensorflow torch torch-geometric gdown')
except Exception as e:
    print(f"Could not automatically run pip install: {e}")

import os
import glob
import json
import time
import copy
import random
import inspect
import platform
import warnings
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cosine
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.metrics import roc_auc_score, average_precision_score

import mne
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
import esinet
from esinet import Simulation, Net

try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

warnings.filterwarnings("default")
plt.switch_backend("agg")
mne.set_log_level("WARNING")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

CFG = {
    "seed": 42,
    "mne_root": "/content/drive/MyDrive/Esinet/mne_data",
    "output_dir": "/content/drive/MyDrive/Esinet/final_convdip_graph_comparison",
    "subject": "sample",
    "spacing": "ico3",
    "epoch_tmin": -0.20,
    "epoch_tmax": 0.50,
    "inference_tmin": 0.00,
    "inference_tmax": 0.29,
    "sfreq": 100.0,
    "n_train": 10000,
    "n_val": 1000,
    "n_test": 2000,
    "duration": 0.30,
    "n_sources": 3,
    "extents": (10, 20),
    "snr": 3.0,
    "hidden": 24,
    "dropout": 0.20,
    "batch_size": 16,
    "epochs": 120,
    "convdip_epochs": 120,
    "convdip_batch": 256,
    "lr": 2e-3,
    "weight_decay": 1e-5,
    "active_ratio": 0.10,
    "active_weight": 20.0,
    "sensor_weight": 0.10,
    "graph_weight": 1e-5,
    "patience": 18,
    "scheduler_patience": 6,
    "tikhonov_relative": 1e-2,
    "split_half_repeats": 20,
}

SEED = CFG["seed"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_all(SEED)
for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as exc:
        print("TensorFlow memory-growth warning:", exc)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

if IN_COLAB:
    drive.mount("/content/drive")

OUT = Path(CFG["output_dir"])
SIM_DIR = OUT / "simulation"
REAL_DIR = OUT / "real_eeg"
MODEL_DIR = OUT / "models"
STC_DIR = OUT / "stc"
FIG_DIR = OUT / "figures"
LOG_DIR = OUT / "logs"
for folder in [OUT, SIM_DIR, REAL_DIR, MODEL_DIR, STC_DIR, FIG_DIR, LOG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

GRAPH_PATH = MODEL_DIR / "best_graph.pt"

# =============================================================================
# 2. FIND AND PREPROCESS MNE SAMPLE DATA
# =============================================================================


def find_first(patterns):
    hits = []
    for pattern in patterns:
        hits.extend(glob.glob(pattern, recursive=True))
    hits = sorted(set(hits))
    return hits[0] if hits else None


def find_subjects_dir(root):
    candidates = glob.glob(os.path.join(root, "**", "subjects"), recursive=True)
    candidates.append(root)
    for candidate in candidates:
        surf = os.path.join(candidate, "sample", "surf")
        mri = os.path.join(candidate, "sample", "mri")
        if os.path.isdir(surf) and os.path.isdir(mri):
            return candidate
    return None


ROOT = CFG["mne_root"]
raw_file = find_first([
    os.path.join(ROOT, "**", "sample_audvis_filt-0-40_raw.fif"),
    os.path.join(ROOT, "**", "sample_audvis_raw.fif"),
])
event_file = find_first([os.path.join(ROOT, "**", "*audvis*-eve.fif")])
trans_file = find_first([
    os.path.join(ROOT, "**", "sample_audvis_raw-trans.fif"),
    os.path.join(ROOT, "**", "*-trans.fif"),
])
bem_file = find_first([
    os.path.join(ROOT, "**", "sample-5120-5120-5120-bem-sol.fif"),
    os.path.join(ROOT, "**", "*-bem-sol.fif"),
])
subjects_dir = find_subjects_dir(ROOT)

for name, value in {
    "raw": raw_file,
    "trans": trans_file,
    "bem": bem_file,
    "subjects_dir": subjects_dir,
}.items():
    if value is None:
        raise FileNotFoundError(f"Missing {name} under {ROOT}")
    print(name, value)

os.environ["SUBJECTS_DIR"] = subjects_dir
mne.set_config("SUBJECTS_DIR", subjects_dir, set_env=True)

raw_full = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)
raw = raw_full.copy().pick(picks=["eeg", "eog", "stim"], exclude="bads")
raw.apply_proj()
raw.filter(1.0, 40.0, picks="eeg", method="fir", phase="zero", verbose=False)
raw.set_eeg_reference("average", projection=True, verbose=False)
raw.apply_proj()

try:
    events = mne.find_events(
        raw_full,
        stim_channel="STI 014",
        shortest_event=1,
        verbose=False,
    )
except Exception as exc:
    if event_file is None:
        raise RuntimeError("Event extraction failed and no event file exists") from exc
    events = mne.read_events(event_file)
    events = events[
        (events[:, 0] >= raw.first_samp)
        & (events[:, 0] <= raw.last_samp)
    ]

candidate_ids = {
    "Auditory_Left": 1,
    "Auditory_Right": 2,
    "Visual_Left": 3,
    "Visual_Right": 4,
}
event_id = {
    name: code
    for name, code in candidate_ids.items()
    if np.any(events[:, 2] == code)
}
if not event_id:
    raise ValueError("No requested auditory/visual events were found")

picks = mne.pick_types(
    raw.info,
    meg=False,
    eeg=True,
    eog=False,
    stim=False,
    exclude="bads",
)
epochs = mne.Epochs(
    raw=raw,
    events=events,
    event_id=event_id,
    tmin=CFG["epoch_tmin"],
    tmax=CFG["epoch_tmax"],
    baseline=(CFG["epoch_tmin"], 0.0),
    picks=picks,
    reject={"eeg": 150e-6},
    preload=True,
    proj=True,
    detrend=1,
    reject_by_annotation=True,
    on_missing="warn",
    verbose=False,
)
epochs.resample(CFG["sfreq"], npad="auto", verbose=False)


def make_evokeds(epoch_object):
    result = {}
    for name in event_id:
        if len(epoch_object[name]) > 0:
            result[name] = (
                epoch_object[name]
                .average()
                .crop(
                    CFG["inference_tmin"],
                    CFG["inference_tmax"],
                    include_tmax=True,
                )
            )
    for prefix in ["Auditory", "Visual"]:
        left = f"{prefix}_Left"
        right = f"{prefix}_Right"
        if left in result and right in result:
            result[f"{prefix}_Average"] = mne.combine_evoked(
                [result[left], result[right]],
                weights="equal",
            )
    return result


evokeds = make_evokeds(epochs)

# =============================================================================
# 3. SUBJECT-SPECIFIC FORWARD MODEL
# =============================================================================

src = mne.setup_source_space(
    CFG["subject"],
    spacing=CFG["spacing"],
    subjects_dir=subjects_dir,
    add_dist=False,
    verbose=False,
)
fwd_free = mne.make_forward_solution(
    epochs.info,
    trans_file,
    src,
    bem_file,
    meg=False,
    eeg=True,
    mindist=5.0,
    n_jobs=1,
    verbose=False,
)
fwd_free = mne.pick_types_forward(
    fwd_free,
    meg=False,
    eeg=True,
    ref_meg=False,
    exclude="bads",
)

common = [
    channel
    for channel in epochs.ch_names
    if channel in fwd_free["info"]["ch_names"]
]
if len(common) < 10:
    raise ValueError("Too few common EEG channels")

epochs.pick(common)
for evoked in evokeds.values():
    evoked.pick(common)

fwd_free = mne.pick_channels_forward(
    fwd_free,
    include=common,
    ordered=True,
    copy=True,
)
order = fwd_free["info"]["ch_names"]
epochs.reorder_channels(order)
for evoked in evokeds.values():
    evoked.reorder_channels(order)

assert epochs.ch_names == fwd_free["info"]["ch_names"]

fwd_fixed = mne.convert_forward_solution(
    fwd_free,
    surf_ori=True,
    force_fixed=True,
    use_cps=True,
    copy=True,
    verbose=False,
)
L = np.asarray(fwd_fixed["sol"]["data"], dtype=np.float64)
n_channels, n_vertices = L.shape
vertices = [
    fwd_fixed["src"][0]["vertno"],
    fwd_fixed["src"][1]["vertno"],
]
coords_mm = np.vstack([
    part["rr"][part["vertno"]]
    for part in fwd_fixed["src"]
]) * 1000.0

# =============================================================================
# 4. CORTICAL GRAPH
# =============================================================================


def cortical_edges(forward):
    edges = set()
    offset = 0
    for hemi in forward["src"]:
        used = np.asarray(hemi["vertno"], dtype=np.int64)
        lookup = {
            int(vertex): local
            for local, vertex in enumerate(used)
        }
        triangles = hemi.get("use_tris", None)
        if triangles is None:
            triangles = hemi.get("tris", None)
        if triangles is None:
            raise ValueError("No cortical mesh triangles are available")

        for triangle in triangles:
            triangle = [int(value) for value in triangle]
            if not all(value in lookup for value in triangle):
                continue
            nodes = [lookup[value] + offset for value in triangle]
            for first, second in [
                (nodes[0], nodes[1]),
                (nodes[1], nodes[2]),
                (nodes[2], nodes[0]),
            ]:
                if first != second:
                    edges.add(tuple(sorted((first, second))))
        offset += len(used)

    edge_array = np.asarray(sorted(edges), dtype=np.int64)
    if edge_array.size == 0:
        raise ValueError("The cortical graph contains no edges")
    return edge_array.T


edge_np = cortical_edges(fwd_fixed)


def normalized_adjacency(edge_index, number_of_nodes, target_device):
    source_nodes, target_nodes = edge_index
    self_nodes = np.arange(number_of_nodes, dtype=np.int64)
    rows = np.concatenate([source_nodes, target_nodes, self_nodes])
    columns = np.concatenate([target_nodes, source_nodes, self_nodes])
    degree = np.maximum(
        np.bincount(rows, minlength=number_of_nodes).astype(np.float32),
        1.0,
    )
    values = (
        1.0 / np.sqrt(degree[rows] * degree[columns])
    ).astype(np.float32)
    return torch.sparse_coo_tensor(
        torch.tensor(
            np.vstack([rows, columns]),
            dtype=torch.long,
            device=target_device,
        ),
        torch.tensor(values, dtype=torch.float32, device=target_device),
        (number_of_nodes, number_of_nodes),
        device=target_device,
    ).coalesce()


adjacency = normalized_adjacency(edge_np, n_vertices, device)
edge_tensor = torch.tensor(edge_np, dtype=torch.long, device=device)
edge_lengths = np.linalg.norm(
    coords_mm[edge_np[0]] - coords_mm[edge_np[1]],
    axis=1,
)
geodesic_graph = coo_matrix(
    (
        np.concatenate([edge_lengths, edge_lengths]),
        (
            np.concatenate([edge_np[0], edge_np[1]]),
            np.concatenate([edge_np[1], edge_np[0]]),
        ),
    ),
    shape=(n_vertices, n_vertices),
).tocsr()

# =============================================================================
# 5. TIKHONOV INITIALIZATION
# =============================================================================


def make_tikhonov_operator(matrix, relative_regularization):
    gram = matrix @ matrix.T
    regularization = (
        relative_regularization
        * np.trace(gram)
        / gram.shape[0]
    )
    operator = matrix.T @ np.linalg.solve(
        gram + regularization * np.eye(gram.shape[0]),
        np.eye(gram.shape[0]),
    )
    return operator.astype(np.float32), float(regularization)


K, effective_regularization = make_tikhonov_operator(
    L,
    CFG["tikhonov_relative"],
)


def apply_tikhonov(eeg):
    eeg = np.asarray(eeg, dtype=np.float32)
    if eeg.ndim == 2:
        return np.einsum("vc,ct->vt", K, eeg, optimize=True).astype(np.float32)
    if eeg.ndim == 3:
        return np.einsum("vc,bct->bvt", K, eeg, optimize=True).astype(np.float32)
    raise ValueError(f"Expected 2-D or 3-D EEG, got {eeg.shape}")

# =============================================================================
# 6. SYNTHETIC TRAIN, VALIDATION, AND TEST SETS
# =============================================================================

simulation_settings = {
    "duration_of_trial": CFG["duration"],
    "number_of_sources": CFG["n_sources"],
    "extents": CFG["extents"],
    "target_snr": CFG["snr"],
}


def simulate(number_of_samples, seed):
    seed_all(seed)
    simulation = Simulation(
        fwd_fixed,
        epochs.info.copy(),
        settings=copy.deepcopy(simulation_settings),
    )
    simulation.simulate(n_samples=number_of_samples)
    return simulation


def as_array_collection(data):
    if isinstance(data, (list, tuple)):
        items = list(data)
    else:
        items = [data[index] for index in range(len(data))]

    arrays = []
    for item in items:
        if hasattr(item, "get_data"):
            array = np.asarray(item.get_data())
        elif hasattr(item, "data"):
            array = np.asarray(item.data)
        else:
            array = np.asarray(item)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2:
            raise ValueError(f"Expected a 2-D item, got {array.shape}")
        arrays.append(array.astype(np.float32, copy=False))
    return np.stack(arrays)


sim_train = simulate(CFG["n_train"], SEED)
sim_val = simulate(CFG["n_val"], SEED + 1)
sim_test = simulate(CFG["n_test"], SEED + 2)

X_eeg_train = as_array_collection(sim_train.eeg_data)
Y_source_train = as_array_collection(sim_train.source_data)
X_eeg_val = as_array_collection(sim_val.eeg_data)
Y_source_val = as_array_collection(sim_val.source_data)
X_eeg_test = as_array_collection(sim_test.eeg_data)
Y_source_test = as_array_collection(sim_test.source_data)

for eeg, source, name in [
    (X_eeg_train, Y_source_train, "train"),
    (X_eeg_val, Y_source_val, "validation"),
    (X_eeg_test, Y_source_test, "test"),
]:
    if eeg.shape[1] != n_channels:
        raise ValueError(f"{name}: channel mismatch {eeg.shape}")
    if source.shape[1] != n_vertices:
        raise ValueError(f"{name}: vertex mismatch {source.shape}")
    if eeg.shape[-1] != source.shape[-1]:
        raise ValueError(f"{name}: time mismatch")

n_times = X_eeg_train.shape[-1]
for name, evoked in evokeds.items():
    if evoked.data.shape[-1] != n_times:
        raise ValueError(
            f"Time mismatch for {name}: real={evoked.data.shape[-1]}, "
            f"synthetic={n_times}. Adjust inference_tmax or duration."
        )

X_init_train_raw = apply_tikhonov(X_eeg_train)
X_init_val_raw = apply_tikhonov(X_eeg_val)
X_init_test_raw = apply_tikhonov(X_eeg_test)


def robust_scale(array):
    return float(max(np.percentile(np.abs(array), 99.5), 1e-12))


X_SCALE = robust_scale(X_init_train_raw)
Y_SCALE = robust_scale(Y_source_train)
EEG_SCALE = robust_scale(X_eeg_train)


def scale_array(array, scale_value):
    return np.clip(array / scale_value, -10.0, 10.0).astype(np.float32)


X_init_train = scale_array(X_init_train_raw, X_SCALE)
X_init_val = scale_array(X_init_val_raw, X_SCALE)
X_init_test = scale_array(X_init_test_raw, X_SCALE)
Y_train_scaled = scale_array(Y_source_train, Y_SCALE)
Y_val_scaled = scale_array(Y_source_val, Y_SCALE)
X_eeg_train_scaled = scale_array(X_eeg_train, EEG_SCALE)
X_eeg_val_scaled = scale_array(X_eeg_val, EEG_SCALE)

# Maps scaled sources to scaled EEG.
L_scaled_tensor = torch.tensor(
    L * Y_SCALE / EEG_SCALE,
    dtype=torch.float32,
    device=device,
)

# =============================================================================
# 7. VERIFIED CONVDIP
# =============================================================================


def build_verified_convdip(forward):
    """Do not silently label ESINet's default architecture as ConvDip."""
    signature = inspect.signature(Net.__init__)
    attempts = []
    candidate_keywords = [
        "model_type",
        "model",
        "architecture",
        "network_type",
        "net_type",
    ]
    for keyword in candidate_keywords:
        if keyword not in signature.parameters:
            continue
        for value in ["convdip", "ConvDip", "CONVDIP"]:
            try:
                model = Net(forward, **{keyword: value})
                return model, {
                    "keyword": keyword,
                    "value": value,
                    "signature": str(signature),
                    "attempts": attempts,
                }
            except Exception as exc:
                attempts.append([keyword, value, str(exc)])

    raise RuntimeError(
        "The installed ESINet version does not expose an explicit ConvDip "
        "selector in Net.__init__. Do not use Net(fwd) and call it ConvDip. "
        f"Signature: {signature}. Attempts: {attempts}"
    )


convdip_model, convdip_audit = build_verified_convdip(fwd_fixed)
(LOG_DIR / "convdip_audit.json").write_text(
    json.dumps(convdip_audit, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

convdip_start = time.perf_counter()
convdip_model.fit(
    sim_train,
    epochs=CFG["convdip_epochs"],
    batch_size=CFG["convdip_batch"],
)
convdip_training_minutes = (time.perf_counter() - convdip_start) / 60.0
try:
    convdip_model.save(str(MODEL_DIR / "convdip"))
except Exception as exc:
    print("ConvDip save warning:", exc)

# =============================================================================
# 8. PROPOSED TIME-RESOLVED RESIDUAL GRAPH NETWORK
# =============================================================================

class ResidualGraphNet(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden, 5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.1),
        )
        self.graph_1 = nn.Linear(hidden, hidden)
        self.graph_2 = nn.Linear(hidden, hidden)
        self.norm_1 = nn.LayerNorm(hidden)
        self.norm_2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden, 1, 1),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    @staticmethod
    def propagate(features, adjacency_matrix):
        batch, vertices_count, times, channels = features.shape
        matrix = (
            features
            .permute(1, 0, 2, 3)
            .reshape(vertices_count, batch * times * channels)
        )
        propagated = torch.sparse.mm(adjacency_matrix, matrix)
        return (
            propagated
            .reshape(vertices_count, batch, times, channels)
            .permute(1, 0, 2, 3)
        )

    def forward(self, initial, adjacency_matrix):
        batch, vertices_count, times = initial.shape
        hidden = self.encoder(
            initial.reshape(batch * vertices_count, 1, times)
        )
        hidden = (
            hidden
            .reshape(batch, vertices_count, -1, times)
            .permute(0, 1, 3, 2)
        )
        propagated = self.graph_1(self.propagate(hidden, adjacency_matrix))
        hidden = self.dropout(
            F.leaky_relu(self.norm_1(hidden + propagated), 0.1)
        )
        propagated = self.graph_2(self.propagate(hidden, adjacency_matrix))
        hidden = self.dropout(
            F.leaky_relu(self.norm_2(hidden + propagated), 0.1)
        )
        decoder_input = (
            hidden
            .permute(0, 1, 3, 2)
            .reshape(batch * vertices_count, -1, times)
        )
        residual = self.decoder(decoder_input).reshape(
            batch,
            vertices_count,
            times,
        )
        return initial + residual


class RefinementDataset(Dataset):
    def __init__(self, initial, target, eeg):
        self.initial = torch.from_numpy(initial)
        self.target = torch.from_numpy(target)
        self.eeg = torch.from_numpy(eeg)

    def __len__(self):
        return len(self.initial)

    def __getitem__(self, index):
        return self.initial[index], self.target[index], self.eeg[index]


graph_model = ResidualGraphNet(CFG["hidden"], CFG["dropout"]).to(device)
train_loader = DataLoader(
    RefinementDataset(X_init_train, Y_train_scaled, X_eeg_train_scaled),
    batch_size=CFG["batch_size"],
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    RefinementDataset(X_init_val, Y_val_scaled, X_eeg_val_scaled),
    batch_size=CFG["batch_size"],
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)
optimizer = optim.AdamW(
    graph_model.parameters(),
    lr=CFG["lr"],
    weight_decay=CFG["weight_decay"],
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.5,
    patience=CFG["scheduler_patience"],
    min_lr=1e-6,
)


def total_loss(prediction, target, eeg):
    target_envelope = target.abs().amax(dim=2)
    maximum = target_envelope.amax(dim=1, keepdim=True).clamp_min(1e-8)
    active = (
        target_envelope >= CFG["active_ratio"] * maximum
    ).float()
    weights = 1.0 + (CFG["active_weight"] - 1.0) * active
    pointwise = F.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
    )
    source_loss = (
        weights.unsqueeze(-1) * pointwise
    ).sum() / (
        weights.sum() * target.shape[-1]
    ).clamp_min(1.0)

    predicted_eeg = torch.einsum(
        "cv,bvt->bct",
        L_scaled_tensor,
        prediction,
    )
    sensor_loss = F.mse_loss(predicted_eeg, eeg)

    spatial_loss = (
        prediction[:, edge_tensor[0], :]
        - prediction[:, edge_tensor[1], :]
    ).pow(2).mean()

    total = (
        source_loss
        + CFG["sensor_weight"] * sensor_loss
        + CFG["graph_weight"] * spatial_loss
    )
    return total, source_loss.detach(), sensor_loss.detach(), spatial_loss.detach()


def run_epoch(loader, training):
    graph_model.train(training)
    sums = np.zeros(4, dtype=float)
    sample_count = 0
    context = nullcontext() if training else torch.no_grad()

    with context:
        for initial, target, eeg in loader:
            initial = initial.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            eeg = eeg.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            prediction = graph_model(initial, adjacency)
            losses = total_loss(prediction, target, eeg)

            if training:
                losses[0].backward()
                torch.nn.utils.clip_grad_norm_(graph_model.parameters(), 1.0)
                optimizer.step()

            batch = initial.shape[0]
            sums += np.array([value.item() for value in losses]) * batch
            sample_count += batch

    return sums / sample_count


history = []
best_loss = np.inf
stale_epochs = 0
graph_start = time.perf_counter()

for epoch in range(1, CFG["epochs"] + 1):
    train_statistics = run_epoch(train_loader, True)
    val_statistics = run_epoch(val_loader, False)
    scheduler.step(val_statistics[0])

    history.append([
        epoch,
        *train_statistics,
        *val_statistics,
        optimizer.param_groups[0]["lr"],
    ])

    if val_statistics[0] < best_loss - 1e-7:
        best_loss = val_statistics[0]
        stale_epochs = 0
        torch.save(
            {
                "state": graph_model.state_dict(),
                "epoch": epoch,
                "val_loss": float(best_loss),
                "cfg": CFG,
                "scales": [X_SCALE, Y_SCALE, EEG_SCALE],
                "channels": epochs.ch_names,
                "vertices": vertices,
            },
            GRAPH_PATH,
        )
    else:
        stale_epochs += 1

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:03d}: "
            f"train={train_statistics[0]:.6f}, "
            f"val={val_statistics[0]:.6f}"
        )

    if stale_epochs >= CFG["patience"]:
        print("Early stopping at epoch", epoch)
        break

graph_training_minutes = (time.perf_counter() - graph_start) / 60.0
history_df = pd.DataFrame(
    history,
    columns=[
        "Epoch",
        "Train total",
        "Train source",
        "Train sensor",
        "Train graph",
        "Val total",
        "Val source",
        "Val sensor",
        "Val graph",
        "Learning rate",
    ],
)
history_df.to_csv(SIM_DIR / "training_history.csv", index=False)
checkpoint = torch.load(GRAPH_PATH, map_location=device, weights_only=False)
graph_model.load_state_dict(checkpoint["state"])
graph_model.eval()

# =============================================================================
# 9. dSPM AND COMMON PREDICTION FUNCTIONS
# =============================================================================

noise_covariance = mne.compute_covariance(
    epochs,
    tmin=CFG["epoch_tmin"],
    tmax=0.0,
    method=["shrunk", "empirical"],
    rank=None,
    verbose=False,
)
dspm_inverse = mne.minimum_norm.make_inverse_operator(
    epochs.info,
    fwd_free,
    noise_covariance,
    loose=0.2,
    depth=0.8,
    fixed=False,
    rank=None,
    verbose=False,
)
lambda2 = 1.0 / CFG["snr"] ** 2


def make_stc(data, times):
    data = np.asarray(data)
    if data.shape != (n_vertices, len(times)):
        raise ValueError(f"Invalid STC shape: {data.shape}")
    tstep = float(times[1] - times[0]) if len(times) > 1 else 1.0
    return mne.SourceEstimate(
        data,
        vertices,
        float(times[0]),
        tstep,
        subject=CFG["subject"],
    )


@torch.no_grad()
def graph_predict(eeg):
    start = time.perf_counter()
    initial = apply_tikhonov(eeg)
    model_input = (
        torch.from_numpy(scale_array(initial, X_SCALE))
        .unsqueeze(0)
        .to(device)
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prediction = graph_model(model_input, adjacency)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prediction = prediction.squeeze(0).cpu().numpy() * Y_SCALE
    elapsed = (time.perf_counter() - start) * 1000.0
    return prediction.astype(np.float32), elapsed


def tikhonov_predict(eeg):
    start = time.perf_counter()
    result = apply_tikhonov(eeg)
    return result, (time.perf_counter() - start) * 1000.0


def convdip_predict(evoked):
    start = time.perf_counter()
    try:
        prediction = convdip_model.predict(evoked)
    except Exception:
        prediction = convdip_model.predict([evoked])
    elapsed = (time.perf_counter() - start) * 1000.0

    if isinstance(prediction, (list, tuple)):
        if not prediction:
            raise ValueError("ConvDip returned an empty result")
        prediction = prediction[0]
    if hasattr(prediction, "data"):
        array = np.asarray(prediction.data)
    else:
        array = np.asarray(prediction)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (n_vertices, len(evoked.times)):
        raise ValueError(
            f"ConvDip output shape {array.shape} does not match "
            f"expected {(n_vertices, len(evoked.times))}"
        )
    return array.astype(np.float32), elapsed


def dspm_predict(evoked):
    start = time.perf_counter()
    result = mne.minimum_norm.apply_inverse(
        evoked,
        dspm_inverse,
        lambda2=lambda2,
        method="dSPM",
        pick_ori=None,
        verbose=False,
    )
    elapsed = (time.perf_counter() - start) * 1000.0
    return result.data.astype(np.float32), elapsed

# =============================================================================
# 10. METRICS
# =============================================================================


def envelope(data):
    return np.max(np.abs(data), axis=1)


def normalize_map(data):
    return data / (np.max(np.abs(data)) + 1e-12)


def peak(data):
    return int(np.argmax(envelope(data)))


def safe_correlation(first, second, function=pearsonr):
    first = np.ravel(first)
    second = np.ravel(second)
    if np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return np.nan
    result = function(first, second)
    return float(
        result.statistic if hasattr(result, "statistic") else result[0]
    )


def sensor_residual(eeg, source):
    return float(
        np.linalg.norm(eeg - L @ source)
        / (np.linalg.norm(eeg) + 1e-12)
    )


def euclidean_error(reference, prediction):
    return float(
        np.linalg.norm(
            coords_mm[peak(reference)]
            - coords_mm[peak(prediction)]
        )
    )


def geodesic_error(reference, prediction):
    distances = dijkstra(
        geodesic_graph,
        directed=False,
        indices=peak(reference),
        return_predecessors=False,
    )
    distance = distances[peak(prediction)]
    return float(distance) if np.isfinite(distance) else np.nan


def synthetic_metrics(reference, prediction, eeg):
    if reference.shape != prediction.shape:
        raise ValueError(
            f"Prediction shape {prediction.shape} != target {reference.shape}"
        )
    reference_map = envelope(reference)
    prediction_map = envelope(prediction)
    labels = (
        reference_map >= 0.10 * np.max(reference_map)
    ).astype(int)

    if np.unique(labels).size == 2:
        auc = roc_auc_score(labels, normalize_map(prediction_map))
        average_precision = average_precision_score(
            labels,
            normalize_map(prediction_map),
        )
    else:
        auc = np.nan
        average_precision = np.nan

    return {
        "MSE": float(np.mean((reference - prediction) ** 2)),
        "Normalized MSE": float(np.mean(
            (normalize_map(reference) - normalize_map(prediction)) ** 2
        )),
        "Source Pearson": safe_correlation(reference, prediction),
        "Spatial Spearman": safe_correlation(
            reference_map,
            prediction_map,
            spearmanr,
        ),
        "ROC AUC": float(auc),
        "Average precision": float(average_precision),
        "Euclidean peak error (mm)": euclidean_error(reference, prediction),
        "Geodesic peak error (mm)": geodesic_error(reference, prediction),
        "Sensor residual": sensor_residual(eeg, prediction),
    }

# =============================================================================
# 11. INDEPENDENT SYNTHETIC TEST
# =============================================================================

synthetic_rows = []
for index in range(CFG["n_test"]):
    eeg = X_eeg_test[index]
    truth = Y_source_test[index]
    synthetic_evoked = mne.EvokedArray(
        eeg.astype(float),
        epochs.info.copy(),
        tmin=0.0,
        nave=1,
        verbose=False,
    )
    outputs = {
        "Tikhonov": tikhonov_predict(eeg),
        "Proposed Graph": graph_predict(eeg),
        "ConvDip": convdip_predict(synthetic_evoked),
        "dSPM": dspm_predict(synthetic_evoked),
    }
    for algorithm, (prediction, elapsed) in outputs.items():
        row = synthetic_metrics(truth, prediction, eeg)
        row.update({
            "Sample": index,
            "Algorithm": algorithm,
            "Inference time (ms)": elapsed,
        })
        synthetic_rows.append(row)

    if (index + 1) % 100 == 0:
        print("Synthetic test", index + 1, "/", CFG["n_test"])

synthetic_df = pd.DataFrame(synthetic_rows)
synthetic_df.to_csv(SIM_DIR / "sample_metrics.csv", index=False)
synthetic_summary = synthetic_df.groupby("Algorithm").agg([
    "mean",
    "std",
    "median",
])
synthetic_summary.to_csv(SIM_DIR / "summary.csv")

# =============================================================================
# 12. REAL EEG INFERENCE AND dSPM AGREEMENT
# =============================================================================

real_outputs = {}
real_rows = []

for condition, evoked in evokeds.items():
    predictions = {
        "Tikhonov": tikhonov_predict(evoked.data),
        "Proposed Graph": graph_predict(evoked.data),
        "ConvDip": convdip_predict(evoked),
        "dSPM": dspm_predict(evoked),
    }
    real_outputs[condition] = {}
    dspm_data = predictions["dSPM"][0]

    for algorithm, (data, elapsed) in predictions.items():
        stc = make_stc(data, evoked.times)
        real_outputs[condition][algorithm] = data
        stc.save(
            str(STC_DIR / f"{algorithm.replace(' ', '_')}_{condition}"),
            overwrite=True,
        )

        reference_map = envelope(dspm_data)
        predicted_map = envelope(data)
        real_rows.append({
            "Condition": condition,
            "Algorithm": algorithm,
            "Spearman agreement with dSPM": safe_correlation(
                reference_map,
                predicted_map,
                spearmanr,
            ),
            "Cosine agreement with dSPM": float(
                1.0 - cosine(
                    normalize_map(reference_map),
                    normalize_map(predicted_map),
                )
            ),
            "Peak distance from dSPM (mm)": euclidean_error(
                dspm_data,
                data,
            ),
            "Sensor residual": sensor_residual(evoked.data, data),
            "Inference time (ms)": elapsed,
            "Interpretation": "dSPM agreement is not ground-truth accuracy",
        })

real_df = pd.DataFrame(real_rows)
real_df.to_csv(REAL_DIR / "real_metrics.csv", index=False)

# =============================================================================
# 13. SPLIT-HALF RELIABILITY ON REAL EEG
# =============================================================================


def predict_named(name, evoked):
    if name == "Tikhonov":
        return tikhonov_predict(evoked.data)[0]
    if name == "Proposed Graph":
        return graph_predict(evoked.data)[0]
    if name == "ConvDip":
        return convdip_predict(evoked)[0]
    if name == "dSPM":
        return dspm_predict(evoked)[0]
    raise ValueError(name)


split_rows = []
rng = np.random.default_rng(SEED + 100)

for condition in event_id:
    condition_epochs = epochs[condition]
    if len(condition_epochs) < 4:
        continue

    for repeat in range(CFG["split_half_repeats"]):
        indices = rng.permutation(len(condition_epochs))
        half = len(indices) // 2
        first = (
            condition_epochs[indices[:half]]
            .average()
            .crop(
                CFG["inference_tmin"],
                CFG["inference_tmax"],
                include_tmax=True,
            )
        )
        second = (
            condition_epochs[indices[half:2 * half]]
            .average()
            .crop(
                CFG["inference_tmin"],
                CFG["inference_tmax"],
                include_tmax=True,
            )
        )

        for algorithm in [
            "Tikhonov",
            "Proposed Graph",
            "ConvDip",
            "dSPM",
        ]:
            first_data = predict_named(algorithm, first)
            second_data = predict_named(algorithm, second)
            split_rows.append({
                "Condition": condition,
                "Repeat": repeat,
                "Algorithm": algorithm,
                "Spatial Spearman reliability": safe_correlation(
                    envelope(first_data),
                    envelope(second_data),
                    spearmanr,
                ),
                "Source Pearson reliability": safe_correlation(
                    first_data,
                    second_data,
                ),
                "Peak distance between halves (mm)": euclidean_error(
                    first_data,
                    second_data,
                ),
            })

split_df = pd.DataFrame(split_rows)
split_df.to_csv(REAL_DIR / "split_half.csv", index=False)

# =============================================================================
# 14. FIGURES AND AUDIT FILES
# =============================================================================

figure, axis = plt.subplots(figsize=(9, 5))
axis.plot(history_df["Epoch"], history_df["Train total"], label="Train")
axis.plot(history_df["Epoch"], history_df["Val total"], label="Validation")
axis.set_xlabel("Epoch")
axis.set_ylabel("Loss")
axis.set_title("Graph training history")
axis.grid(alpha=0.3)
axis.legend()
figure.tight_layout()
figure.savefig(FIG_DIR / "training_history.png", dpi=300)
plt.close(figure)

figure, axes = plt.subplots(2, 2, figsize=(15, 10))
for axis, metric in zip(
    axes.flat,
    [
        "Normalized MSE",
        "ROC AUC",
        "Geodesic peak error (mm)",
        "Sensor residual",
    ],
):
    sns.boxplot(
        data=synthetic_df,
        x="Algorithm",
        y=metric,
        showfliers=False,
        ax=axis,
    )
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.3)
figure.suptitle("Independent synthetic test with known ground truth")
figure.tight_layout()
figure.savefig(FIG_DIR / "synthetic_comparison.png", dpi=300)
plt.close(figure)

if not split_df.empty:
    figure, axis = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=split_df,
        x="Condition",
        y="Spatial Spearman reliability",
        hue="Algorithm",
        showfliers=False,
        ax=axis,
    )
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(FIG_DIR / "split_half.png", dpi=300)
    plt.close(figure)

serializable_cfg = copy.deepcopy(CFG)
serializable_cfg["extents"] = list(serializable_cfg["extents"])

(OUT / "configuration.json").write_text(
    json.dumps(
        {
            **serializable_cfg,
            "raw_file": raw_file,
            "trans_file": trans_file,
            "bem_file": bem_file,
            "subjects_dir": subjects_dir,
            "channels": n_channels,
            "vertices": n_vertices,
            "edges": int(edge_np.shape[1]),
            "effective_regularization": effective_regularization,
            "scales": {
                "input": X_SCALE,
                "target": Y_SCALE,
                "eeg": EEG_SCALE,
            },
            "best_graph_epoch": int(checkpoint["epoch"]),
            "convdip_training_minutes": convdip_training_minutes,
            "graph_training_minutes": graph_training_minutes,
            "convdip_audit": convdip_audit,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

(OUT / "environment.json").write_text(
    json.dumps(
        {
            "python": platform.python_version(),
            "mne": mne.__version__,
            "torch": torch.__version__,
            "tensorflow": tf.__version__,
            "esinet": getattr(esinet, "__version__", "unknown"),
            "device": str(device),
        },
        indent=2,
    ),
    encoding="utf-8",
)

print("\nSynthetic ground-truth summary:\n", synthetic_summary.round(4))
print(
    "\nReal EEG summary (agreement, not ground truth):\n",
    real_df.groupby("Algorithm").mean(numeric_only=True).round(4),
)
print("\nOutputs:", OUT)
print("WARNING: Real-data agreement with dSPM is not localization accuracy.")
