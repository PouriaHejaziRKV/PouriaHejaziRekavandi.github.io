# -*- coding: utf-8 -*-
r"""
CORRECTED PHYSICS-GAT / CONVDIP BENCHMARK PIPELINE
Recommended installation (run separately in Colab):
!pip install -q esinet "mne>=1.7,<1.10" pandas matplotlib \
    scikit-learn scipy torch-geometric psutil

Major corrections in this revision
No HTML-escaped Python operators. Functional graph construction is O(VKT),
not O(V^2*T), by restricting dynamic neighbours to a precomputed candidate set.
Dynamic graphs are cached with a bounded LRU cache for validation/test and
optionally for training; full dense VxV functional matrices are never made.
SNR values are explicitly in dB.
dSPM lambda2 uses linear amplitude SNR.
ConvDip outputs are recursively flattened, vertex order is verified when
metadata are available, and time is interpolated using actual time vectors.
Silent cropping/zero-padding has been removed.
Zero/constant-vector metric cases are handled safely.
Localization uses connected-component centres from the known target support
instead of treating arbitrary map peaks as the simulator's true generators.
Shape-normalized and physical/calibration-sensitive metrics are separated.
Forward consistency is checked before training and can be noise-whitened.
Loss logs raw and weighted terms; residual smoothness is named accurately.
Checkpoint hashes include lead-field, channels, vertices, environment, and
code schema identifiers.
DataLoader generators are explicit and checkpointed.
Timing uses warm-up and repeated measurements and reports median/IQR/p95.
Cross-seed summaries and paired bootstrap differences are written.
Added practical OOD channel corruption and empirical-noise hooks.
True cross-subject/model-mismatch tests still require additional forward models.
Large full-mode arrays can be placed in memory-mapped storage.

Scientific scope
Synthetic data provide ground-truth evaluation.
Real EEG results measure agreement with a same-forward-model dSPM comparator,
not biological accuracy. Parameter shifts and sensor corruption are robustness tests;
physical cross-subject generalization requires extra subjects/forward models.
"""

from __future__ import annotations

try:
    import IPython
    IPython.get_ipython().system('pip install -q esinet "mne>=1.7,<1.10" pandas matplotlib scikit-learn scipy torch-geometric psutil')
except Exception:
    pass

import gc
import glob
import hashlib
import importlib.metadata as metadata
import inspect
import json
import math
import os
import platform
import random
import statistics
import time
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import add_self_loops, coalesce
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import psutil
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from esinet import Net, Simulation
warnings.filterwarnings("default")
plt.switch_backend("agg")
mne.set_log_level("WARNING")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

@dataclass
class Config:
    run_mode: str = "smoke" # smoke | pilot | full
    seeds: tuple[int, ...] = (42, 52, 62, 72, 82)
    mne_root: str = str(Path("~/mne_data/MNE-sample-data").expanduser())
    output_dir: str = "sandbox_data/Esinet/final_physics_gat_corrected"
    memmap_dir: str = "sandbox_data/Esinet/final_physics_gat_corrected/memmap"
    use_memmap_full: bool = True

    source_spacing: str = "ico3"
    duration_s: float = 0.30
    sfreq: float = 100.0
    mindist_mm: float = 5.0

    # ESINet target_snr is treated explicitly as dB in this benchmark.
    train_snr_db: float = 3.0
    ood_snr_db: float = -10.0
    train_extents: tuple[int, int] = (10, 20)
    ood_extents: tuple[int, int] = (28, 42)
    train_sources: int = 3
    ood_sources: int = 6

    # Sensor robustness tests retain the same sensor layout but corrupt values.
    ood_channel_dropout_fraction: float = 0.15
    ood_channel_noise_sd_ratio: float = 0.25

    n_train_full: int = 40_000
    n_validation_full: int = 1_000
    n_test_full: int = 1_000
    epochs_full: int = 150
    batch_size_full: int = 64

    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    scheduler_patience: int = 7
    early_stopping_patience: int = 20

    tikhonov_relative: float = 1e-2
    hidden: int = 32
    heads: int = 4
    dropout: float = 0.15

    # Restricted dynamic graph: O(V * candidate_k * T).
    leadfield_candidate_k: int = 24
    spatial_candidate_k: int = 24
    functional_k: int = 3
    graph_cache_items: int = 512
    cache_training_graphs: bool = False

    active_ratio: float = 0.10
    active_weight: float = 50.0
    residual_smoothness_weight: float = 1e-4
    geodesic_weight: float = 2e-3
    forward_weight: float = 1e-2
    geodesic_temperature_mm: float = 10.0
    whiten_forward_loss: bool = True

    peak_ratio: float = 0.25
    peak_separation_mm: float = 20.0
    maximum_peaks: int = 12
    unmatched_penalty_mm: float = 100.0
    interhemispheric_penalty_mm: float = 300.0

    epoch_tmin: float = -0.20
    epoch_tmax: float = 0.50
    inference_tmin: float = 0.00
    inference_tmax: float = 0.30

    run_real_eeg: bool = True
    train_convdip: bool = True
    resume: bool = True

    timing_warmups: int = 5
    timing_repeats: int = 30
    bootstrap_repeats: int = 2_000

    def resolved(self) -> dict[str, Any]:
        if self.run_mode == "smoke":
            return {
                "seeds": (self.seeds[0],),
                "n_train": 16,
                "n_validation": 4,
                "n_test": 4,
                "epochs": 1,
                "batch_size": 2,
            }
        if self.run_mode == "pilot":
            return {
                "seeds": (self.seeds[0],),
                "n_train": 1_000,
                "n_validation": 200,
                "n_test": 200,
                "epochs": 5,
                "batch_size": 8,
            }
        if self.run_mode == "full":
            return {
                "seeds": self.seeds,
                "n_train": self.n_train_full,
                "n_validation": self.n_validation_full,
                "n_test": self.n_test_full,
                "epochs": self.epochs_full,
                "batch_size": self.batch_size_full,
            }
        raise ValueError("run_mode must be smoke, pilot, or full")

CFG = Config()
RUN = CFG.resolved()
OUT = Path(CFG.output_dir)
MEMMAP = Path(CFG.memmap_dir)
OUT.mkdir(parents=True, exist_ok=True)
MEMMAP.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCHEMA_VERSION = "physics-gat-corrected-v2"

if CFG.hidden % CFG.heads:
    raise ValueError("hidden must be divisible by heads")

# =============================================================================
# 2. UTILITIES
# =============================================================================

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def assert_finite(name: str, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        valid = bool(torch.isfinite(value).all().item())
        shape = tuple(value.shape)
    else:
        array = np.asarray(value)
        valid = bool(np.isfinite(array).all())
        shape = array.shape
    if not valid:
        raise RuntimeError(f"{name} contains non-finite values; shape={shape}")

def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()

def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"

def environment_manifest() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda": torch.version.cuda if hasattr(torch.version, 'cuda') else None,
        "torch": torch.__version__,
        "torch_geometric": package_version("torch-geometric"),
        "mne": mne.__version__,
        "esinet": package_version("esinet"),
        "numpy": np.__version__,
        "scipy": package_version("scipy"),
        "pandas": pd.__version__,
        "scikit_learn": package_version("scikit-learn"),
    }

def robust_scale(array: np.ndarray, percentile: float = 99.5) -> float:
    return max(float(np.percentile(np.abs(array), percentile)), 1e-12)

def snr_db_to_lambda2(snr_db: float) -> float:
    amplitude_snr = 10.0 ** (snr_db / 20.0)
    return 1.0 / max(amplitude_snr * amplitude_snr, 1e-12)

def find_all(patterns: list[str]) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        found.extend(glob.glob(pattern, recursive=True))
    return sorted(set(found))

def find_first(patterns: list[str]) -> Optional[str]:
    found = find_all(patterns)
    return found[0] if found else None

def find_subjects_directory(root: str) -> Optional[str]:
    candidates = find_all([os.path.join(root, "**", "subjects")]) + [root]
    for directory in candidates:
        if (
            os.path.isdir(os.path.join(directory, "sample", "surf")) and
            os.path.isdir(os.path.join(directory, "sample", "mri"))
        ):
            return directory
    return None

def make_memmap(name: str, shape: tuple[int, ...], dtype=np.float32) -> np.memmap:
    path = MEMMAP / f"{name}.dat"
    return np.memmap(path, mode="w+", dtype=dtype, shape=shape)

def maybe_memmap_copy(name: str, value: np.ndarray) -> np.ndarray:
    if CFG.run_mode != "full" or not CFG.use_memmap_full:
        return np.asarray(value, dtype=np.float32)
    output = make_memmap(name, value.shape)
    output[:] = value
    output.flush()
    return output

seed_all(RUN["seeds"][0])
print("Device:", DEVICE)
print("RAM GB:", round(psutil.virtual_memory().total / 1024**3, 2))

# =============================================================================
# 3. MNE DATA AND FORWARD MODEL
# =============================================================================

MNE_ROOT = CFG.mne_root
raw_file = find_first([
    os.path.join(MNE_ROOT, "**", "sample_audvis_filt-0-40_raw.fif"),
    os.path.join(MNE_ROOT, "**", "sample_audvis_raw.fif"),
])
event_file = find_first([
    os.path.join(MNE_ROOT, "**", "sample_audvis_raw-eve.fif"),
    os.path.join(MNE_ROOT, "**", "audvis-eve.fif"),
])
trans_file = find_first([
    os.path.join(MNE_ROOT, "**", "sample_audvis_raw-trans.fif"),
    os.path.join(MNE_ROOT, "**", "audvis-trans.fif"),
])
bem_file = find_first([
    os.path.join(MNE_ROOT, "**", "sample-5120-5120-5120-bem-sol.fif"),
    os.path.join(MNE_ROOT, "**", "*-bem-sol.fif"),
])
subjects_dir = find_subjects_directory(MNE_ROOT)

for key, value in {
    "raw_file": raw_file,
    "event_file": event_file,
    "trans_file": trans_file,
    "bem_file": bem_file,
    "subjects_dir": subjects_dir,
}.items():
    if value is None:
        raise FileNotFoundError(f"{key} not found under {MNE_ROOT}")

os.environ["SUBJECTS_DIR"] = str(subjects_dir)
mne.set_config("SUBJECTS_DIR", str(subjects_dir), set_env=True)

raw_full = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)
raw = raw_full.copy().pick(picks=["eeg", "eog", "stim"], exclude="bads")
raw.apply_proj()
raw.filter(1.0, 40.0, picks="eeg", method="fir", phase="zero", verbose=False)
raw.set_eeg_reference("average", projection=True, verbose=False)
raw.apply_proj()

try:
    events = mne.find_events(raw_full, stim_channel="STI 014", shortest_event=1, verbose=False)
except Exception:
    events = mne.read_events(event_file)
events = events[(events[:, 0] >= raw.first_samp) & (events[:, 0] <= raw.last_samp)]

requested_event_id = {
    "Auditory_Left": 1,
    "Auditory_Right": 2,
    "Visual_Left": 3,
    "Visual_Right": 4,
}
event_id = {name: code for name, code in requested_event_id.items() if np.any(events[:, 2] == code)}
if not event_id:
    raise RuntimeError("No requested events found")

eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude="bads")
epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=CFG.epoch_tmin,
    tmax=CFG.epoch_tmax,
    baseline=(CFG.epoch_tmin, 0.0),
    picks=eeg_picks,
    reject={"eeg": 150e-6},
    preload=True,
    proj=True,
    detrend=1,
    reject_by_annotation=True,
    on_missing="warn",
    verbose=False,
)
epochs.resample(CFG.sfreq, npad="auto", verbose=False)

evoked_conditions = {
    name: epochs[name].average().crop(CFG.inference_tmin, CFG.inference_tmin + (int(round(CFG.duration_s * CFG.sfreq)) - 1) / CFG.sfreq)
    for name in event_id if len(epochs[name]) > 0
}
if {"Auditory_Left", "Auditory_Right"}.issubset(evoked_conditions):
    evoked_conditions["Auditory_Average"] = mne.combine_evoked(
        [evoked_conditions["Auditory_Left"], evoked_conditions["Auditory_Right"]],
        weights="equal",
    )
if {"Visual_Left", "Visual_Right"}.issubset(evoked_conditions):
    evoked_conditions["Visual_Average"] = mne.combine_evoked(
        [evoked_conditions["Visual_Left"], evoked_conditions["Visual_Right"]],
        weights="equal",
    )

del raw_full
gc.collect()

source_space = mne.setup_source_space(
    "sample",
    spacing=CFG.source_spacing,
    subjects_dir=str(subjects_dir),
    add_dist=False,
    verbose=False,
)
fwd_free = mne.make_forward_solution(
    epochs.info,
    trans=str(trans_file),
    src=source_space,
    bem=str(bem_file),
    meg=False,
    eeg=True,
    mindist=CFG.mindist_mm,
    n_jobs=1,
    verbose=False,
)
fwd_free = mne.pick_types_forward(fwd_free, meg=False, eeg=True, ref_meg=False, exclude="bads")
common_channels = [ch for ch in epochs.ch_names if ch in fwd_free["info"]["ch_names"]]
if not common_channels:
    raise RuntimeError("No common EEG channels between data and forward model")

epochs.pick(common_channels)
for evoked in evoked_conditions.values():
    evoked.pick(common_channels)
fwd_free = mne.pick_channels_forward(fwd_free, include=common_channels, ordered=True, copy=True)
forward_channels = list(fwd_free["info"]["ch_names"])
epochs.reorder_channels(forward_channels)
for evoked in evoked_conditions.values():
    evoked.reorder_channels(forward_channels)

fwd_fixed = mne.convert_forward_solution(
    fwd_free,
    surf_ori=True,
    force_fixed=True,
    use_cps=True,
    copy=True,
    verbose=False,
)
lead_field = np.asarray(fwd_fixed["sol"]["data"], dtype=np.float32)
N_CHANNELS, N_VERTICES = lead_field.shape
N_TIMES = int(round(CFG.duration_s * CFG.sfreq))
TARGET_TIMES = np.arange(N_TIMES, dtype=np.float64) / CFG.sfreq
SOURCE_VERTICES = [
    np.asarray(fwd_fixed["src"][0]["vertno"], dtype=np.int64),
    np.asarray(fwd_fixed["src"][1]["vertno"], dtype=np.int64),
]
coordinates_mm = np.vstack([
    src["rr"][src["vertno"]] for src in fwd_fixed["src"]
]).astype(np.float32) * 1000.0

if list(fwd_fixed["sol"]["row_names"]) != forward_channels:
    raise RuntimeError("Forward channel order is inconsistent")

# =============================================================================
# 4. CORTICAL GRAPH, GEODESICS, AND RESTRICTED CANDIDATES
# =============================================================================

def build_cortical_edges(forward_model: Any) -> np.ndarray:
    edge_set: set[tuple[int, int]] = set()
    offset = 0
    for hemisphere in forward_model["src"]:
        used = np.asarray(hemisphere["vertno"], dtype=np.int64)
        lookup = {int(vertex): index for index, vertex in enumerate(used)}
        triangles = hemisphere.get("use_tris", hemisphere.get("tris"))
        if triangles is None:
            raise RuntimeError("No cortical triangles found")
        for triangle in np.asarray(triangles, dtype=np.int64):
            if not all(int(vertex) in lookup for vertex in triangle):
                continue
            nodes = [lookup[int(vertex)] + offset for vertex in triangle]
            for first, second in ((nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])):
                if first != second:
                    edge_set.add(tuple(sorted((first, second))))
        offset += len(used)
    if not edge_set:
        raise RuntimeError("Cortical graph is empty")
    return np.asarray(sorted(edge_set), dtype=np.int64).T

def compute_geodesic_matrix(edges: np.ndarray, xyz_mm: np.ndarray) -> np.ndarray:
    first, second = edges
    lengths = np.linalg.norm(xyz_mm[first] - xyz_mm[second], axis=1)
    graph = coo_matrix(
        (
            np.concatenate([lengths, lengths]),
            (np.concatenate([first, second]), np.concatenate([second, first])),
        ),
        shape=(len(xyz_mm), len(xyz_mm)),
    ).tocsr()
    distances = np.asarray(dijkstra(graph, directed=False), dtype=np.float32)
    distances[~np.isfinite(distances)] = CFG.interhemispheric_penalty_mm
    return distances

UNDIRECTED_EDGES = build_cortical_edges(fwd_fixed)
GEODESIC_MM = compute_geodesic_matrix(UNDIRECTED_EDGES, coordinates_mm)

def make_local_edge_index() -> torch.Tensor:
    first, second = UNDIRECTED_EDGES
    edge_index = torch.tensor(
        np.vstack([
            np.concatenate([first, second]),
            np.concatenate([second, first]),
        ]),
        dtype=torch.long,
    )
    edge_index, _ = add_self_loops(edge_index, num_nodes=N_VERTICES)
    return coalesce(edge_index, num_nodes=N_VERTICES)

LOCAL_EDGE_INDEX = make_local_edge_index()
SPATIAL_EDGE_TENSOR = torch.tensor(
    np.vstack([
        np.concatenate([UNDIRECTED_EDGES[0], UNDIRECTED_EDGES[1]]),
        np.concatenate([UNDIRECTED_EDGES[1], UNDIRECTED_EDGES[0]]),
    ]),
    dtype=torch.long,
    device=DEVICE,
)
GEODESIC_TENSOR = torch.tensor(GEODESIC_MM, dtype=torch.float32, device=DEVICE)
LEAD_FIELD_TENSOR = torch.tensor(lead_field, dtype=torch.float32, device=DEVICE)

def normalized_leadfield_profiles(matrix: np.ndarray) -> np.ndarray:
    profiles = np.asarray(matrix.T, dtype=np.float32).copy()
    profiles -= profiles.mean(axis=1, keepdims=True)
    profiles /= np.linalg.norm(profiles, axis=1, keepdims=True) + 1e-8
    return profiles

def top_k_without_self(scores: np.ndarray, k: int) -> np.ndarray:
    if not 1 <= k < scores.shape[1]:
        raise ValueError("k must be in [1, n-1]")
    scores = scores.copy()
    np.fill_diagonal(scores, -np.inf)
    indices = np.argpartition(scores, -k, axis=1)[:, -k:]
    selected_scores = np.take_along_axis(scores, indices, axis=1)
    order = np.argsort(selected_scores, axis=1)[:, ::-1]
    return np.take_along_axis(indices, order, axis=1).astype(np.int64)

def build_candidate_neighbours(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    profiles = normalized_leadfield_profiles(matrix)
    lf_similarity = np.abs(profiles @ profiles.T)
    lf_neighbours = top_k_without_self(lf_similarity, min(CFG.leadfield_candidate_k, N_VERTICES - 1))

    spatial_scores = -GEODESIC_MM.astype(np.float32)
    spatial_neighbours = top_k_without_self(spatial_scores, min(CFG.spatial_candidate_k, N_VERTICES - 1))

    candidates: list[np.ndarray] = []
    for vertex in range(N_VERTICES):
        merged = np.unique(np.concatenate([lf_neighbours[vertex], spatial_neighbours[vertex]]))
        merged = merged[merged != vertex]
        candidates.append(merged)
    width = max(len(item) for item in candidates)
    padded = np.full((N_VERTICES, width), -1, dtype=np.int64)
    for vertex, item in enumerate(candidates):
        padded[vertex, : len(item)] = item
    return padded, profiles

CANDIDATES, LEAD_PROFILES = build_candidate_neighbours(lead_field)

def create_edge_attributes(
    edge_index: torch.Tensor,
    functional_values: Optional[dict[tuple[int, int], float]] = None,
) -> torch.Tensor:
    first = edge_index[0].cpu().numpy()
    second = edge_index[1].cpu().numpy()
    euclidean = np.linalg.norm(coordinates_mm[first] - coordinates_mm[second], axis=1) / 200.0
    geodesic = GEODESIC_MM[first, second] / max(CFG.interhemispheric_penalty_mm, 1.0)
    signed_lf = np.sum(LEAD_PROFILES[first] * LEAD_PROFILES[second], axis=1)
    abs_lf = np.abs(signed_lf)
    functional = np.zeros(len(first), dtype=np.float32)
    if functional_values:
        functional = np.asarray([
            functional_values.get((int(a), int(b)), 0.0)
            for a, b in zip(first, second)
        ], dtype=np.float32)
    self_mask = first == second
    euclidean[self_mask] = 0.0
    geodesic[self_mask] = 0.0
    signed_lf[self_mask] = 1.0
    abs_lf[self_mask] = 1.0
    functional[self_mask] = 0.0
    values = np.column_stack([euclidean, geodesic, abs_lf, signed_lf, functional]).astype(np.float32)
    assert_finite("edge attributes", values)
    return torch.from_numpy(values)

class DynamicGraphBuilder:
    def __init__(self, cache_items: int):
        source = np.repeat(np.arange(N_VERTICES), CFG.leadfield_candidate_k)
        target = top_k_without_self(
            np.abs(LEAD_PROFILES @ LEAD_PROFILES.T),
            min(CFG.leadfield_candidate_k, N_VERTICES - 1),
        ).reshape(-1)
        source = np.repeat(np.arange(N_VERTICES), target.size // N_VERTICES)
        nonlocal_edges = torch.tensor(np.vstack([source, target]), dtype=torch.long)
        self.base_edges = coalesce(
            torch.cat([LOCAL_EDGE_INDEX, nonlocal_edges, nonlocal_edges.flip(0)], dim=1),
            num_nodes=N_VERTICES,
        )
        self.cache_items = cache_items
        self.cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

    def _key(self, source: np.ndarray) -> str:
        return hashlib.blake2b(np.ascontiguousarray(source).view(np.uint8), digest_size=12).hexdigest()

    def __call__(self, initial_source: np.ndarray, use_cache: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if initial_source.shape != (N_VERTICES, N_TIMES):
            raise ValueError(f"Unexpected source shape {initial_source.shape}")
        key = self._key(initial_source)
        if use_cache and key in self.cache:
            self.cache.move_to_end(key)
            edge_index, edge_attr = self.cache[key]
            return edge_index.clone(), edge_attr.clone()

        signal = initial_source.astype(np.float32, copy=True)
        signal -= signal.mean(axis=1, keepdims=True)
        signal /= np.linalg.norm(signal, axis=1, keepdims=True) + 1e-8

        selected_targets = np.empty((N_VERTICES, CFG.functional_k), dtype=np.int64)
        selected_scores = np.empty((N_VERTICES, CFG.functional_k), dtype=np.float32)
        for vertex in range(N_VERTICES):
            candidates = CANDIDATES[vertex]
            candidates = candidates[candidates >= 0]
            similarities = np.abs(signal[candidates] @ signal[vertex])
            k = min(CFG.functional_k, len(candidates))
            positions = np.argpartition(similarities, -k)[-k:]
            order = positions[np.argsort(similarities[positions])[::-1]]
            chosen = candidates[order]
            values = similarities[order]
            if k < CFG.functional_k:
                chosen = np.pad(chosen, (0, CFG.functional_k - k), constant_values=vertex)
                values = np.pad(values, (0, CFG.functional_k - k))
            selected_targets[vertex] = chosen
            selected_scores[vertex] = values

        source = np.repeat(np.arange(N_VERTICES), CFG.functional_k)
        target = selected_targets.reshape(-1)
        scores = selected_scores.reshape(-1)
        dynamic_edges = torch.tensor(np.vstack([source, target]), dtype=torch.long)
        final_edges = coalesce(
            torch.cat([self.base_edges, dynamic_edges, dynamic_edges.flip(0)], dim=1),
            num_nodes=N_VERTICES,
        )
        values: dict[tuple[int, int], float] = {}
        for a, b, score in zip(source, target, scores):
            values[(int(a), int(b))] = max(values.get((int(a), int(b)), 0.0), float(score))
            values[(int(b), int(a))] = max(values.get((int(b), int(a)), 0.0), float(score))
        attributes = create_edge_attributes(final_edges, values)

        if use_cache and self.cache_items > 0:
            self.cache[key] = (final_edges.clone(), attributes.clone())
            self.cache.move_to_end(key)
            while len(self.cache) > self.cache_items:
                self.cache.popitem(last=False)
        return final_edges, attributes

DYNAMIC_GRAPH_BUILDER = DynamicGraphBuilder(CFG.graph_cache_items)

def make_sparse_adjacency() -> torch.Tensor:
    source, target = UNDIRECTED_EDGES
    self_nodes = np.arange(N_VERTICES, dtype=np.int64)
    rows = np.concatenate([source, target, self_nodes])
    cols = np.concatenate([target, source, self_nodes])
    degree = np.maximum(np.bincount(rows, minlength=N_VERTICES), 1).astype(np.float32)
    values = (1.0 / np.sqrt(degree[rows] * degree[cols])).astype(np.float32)
    return torch.sparse_coo_tensor(
        torch.tensor(np.vstack([rows, cols]), dtype=torch.long, device=DEVICE),
        torch.tensor(values, dtype=torch.float32, device=DEVICE),
        size=(N_VERTICES, N_VERTICES),
        device=DEVICE,
    ).coalesce()

SPARSE_ADJACENCY = make_sparse_adjacency()
# =============================================================================
# 5. DATA EXTRACTION, TIKHONOV, AND FORWARD VALIDATION
# =============================================================================

def extract_array(item: Any) -> np.ndarray:
    if hasattr(item, "get_data"): return np.asarray(item.get_data())
    if hasattr(item, "data"): return np.asarray(item.data)
    return np.asarray(item)

def ensure_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected 2D sample, received {array.shape}")
    return array.astype(np.float32, copy=False)

def extract_collection(data: Any) -> np.ndarray:
    if isinstance(data, np.ndarray):
        if data.ndim == 3: return data.astype(np.float32, copy=False)
        if data.ndim == 2: return data[None].astype(np.float32, copy=False)
        raise ValueError(f"Unexpected collection shape {data.shape}")
    return np.stack([ensure_2d(extract_array(item)) for item in list(data)], axis=0).astype(np.float32)

def make_tikhonov_operator(matrix: np.ndarray) -> np.ndarray:
    matrix64 = np.asarray(matrix, dtype=np.float64)
    gram = matrix64 @ matrix64.T
    regularization = CFG.tikhonov_relative * max(np.trace(gram) / gram.shape[0], 1e-30)
    return np.linalg.solve(gram + regularization * np.eye(gram.shape[0]), matrix64).T.astype(np.float32)

TIKHONOV_OPERATOR = make_tikhonov_operator(lead_field)

def apply_tikhonov(eeg: np.ndarray) -> np.ndarray:
    return np.einsum("vc,bct->bvt", TIKHONOV_OPERATOR, eeg, optimize=True).astype(np.float32)

def validate_forward_consistency(eeg: np.ndarray, source: np.ndarray, label: str) -> dict[str, float]:
    reconstructed = np.einsum("cv,bvt->bct", lead_field, source, optimize=True)
    residual = reconstructed - eeg
    relative = float(np.linalg.norm(residual) / max(np.linalg.norm(eeg), 1e-12))
    explained = float(1.0 - np.sum(residual**2) / max(np.sum(eeg**2), 1e-30))
    result = {"label": label, "relative_residual": relative, "explained_variance": explained}
    if not np.isfinite(relative):
        raise RuntimeError("Forward consistency check failed")
    return result

# =============================================================================
# 6. MODELS
# =============================================================================

class TemporalEncoder(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 8, 5, padding=2),
            nn.GroupNorm(2, 8),
            nn.GELU(),
            nn.Conv1d(8, 12, 3, padding=1),
            nn.GroupNorm(3, 12),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.projection = nn.Linear(48, hidden)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 2:
            raise ValueError("TemporalEncoder expects [nodes, times]")
        return self.projection(self.network(signal[:, None]).flatten(1))

class TikhonovTemporalCNN(nn.Module):
    """Shared vertex-wise temporal CNN applied to Tikhonov source estimates."""
    def __init__(self):
        super().__init__()
        self.encoder = TemporalEncoder(CFG.hidden)
        self.decoder = nn.Sequential(nn.GELU(), nn.Linear(CFG.hidden, N_TIMES))

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        batch, vertices, times = source.shape
        encoded = self.encoder(source.reshape(batch * vertices, times))
        return self.decoder(encoded).reshape(batch, vertices, N_TIMES)

class SparseSpatioTemporalGraphNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TemporalEncoder(CFG.hidden)
        self.projection_1 = nn.Linear(CFG.hidden, CFG.hidden)
        self.projection_2 = nn.Linear(CFG.hidden, CFG.hidden)
        self.norm_1 = nn.LayerNorm(CFG.hidden)
        self.norm_2 = nn.LayerNorm(CFG.hidden)
        self.gate = nn.Sequential(nn.Linear(CFG.hidden, CFG.hidden), nn.Sigmoid())
        self.decoder = nn.Linear(CFG.hidden, N_TIMES)

    @staticmethod
    def propagate(features: torch.Tensor) -> torch.Tensor:
        batch, vertices, channels = features.shape
        matrix = features.permute(1, 0, 2).reshape(vertices, batch * channels)
        propagated = torch.sparse.mm(SPARSE_ADJACENCY, matrix)
        return propagated.reshape(vertices, batch, channels).permute(1, 0, 2)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        batch, vertices, times = source.shape
        hidden = self.encoder(source.reshape(batch * vertices, times)).reshape(batch, vertices, -1)
        hidden = F.gelu(self.norm_1(hidden + self.projection_1(self.propagate(hidden))))
        hidden = hidden * self.gate(hidden)
        hidden = self.norm_2(hidden + self.projection_2(self.propagate(hidden)))
        return self.decoder(F.gelu(hidden))

class EdgeAwareGATBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat = GATv2Conv(
            CFG.hidden,
            CFG.hidden // CFG.heads,
            heads=CFG.heads,
            edge_dim=5,
            dropout=CFG.dropout,
            add_self_loops=False,
        )
        self.norm = nn.LayerNorm(CFG.hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        message = self.gat(x, edge_index, edge_attr)
        message = F.dropout(F.gelu(message), CFG.dropout, self.training)
        return self.norm(x + message)

class FullPhysicsGAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TemporalEncoder(CFG.hidden)
        self.block_1 = EdgeAwareGATBlock()
        self.block_2 = EdgeAwareGATBlock()
        self.decoder = nn.Sequential(
            nn.Linear(CFG.hidden, CFG.hidden),
            nn.GELU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(CFG.hidden, N_TIMES),
        )

    def forward(self, batch_data: Data) -> torch.Tensor:
        counts = torch.bincount(batch_data.batch)
        if not torch.all(counts == N_VERTICES):
            raise RuntimeError("PhysicsGAT requires equal full-size graphs")
        x = self.encoder(batch_data.x)
        x = self.block_1(x, batch_data.edge_index, batch_data.edge_attr)
        x = self.block_2(x, batch_data.edge_index, batch_data.edge_attr)
        return self.decoder(x).reshape(batch_data.batch.max() + 1, N_VERTICES, N_TIMES)

# =============================================================================
# 7. DATASET AND LOADERS
# =============================================================================
class SourceDataset(Dataset):
    def __init__(
        self,
        initial_source: np.ndarray,
        target_source: np.ndarray,
        physical_eeg: np.ndarray,
        build_graph: bool,
        cache_graphs: bool,
    ):
        self.initial_source = initial_source
        self.target_source = target_source
        self.physical_eeg = physical_eeg
        self.build_graph = build_graph
        self.cache_graphs = cache_graphs

    def __len__(self) -> int:
        return len(self.initial_source)

    def __getitem__(self, index: int) -> Data:
        source = np.asarray(self.initial_source[index], dtype=np.float32)
        target = np.asarray(self.target_source[index], dtype=np.float32)
        eeg = np.asarray(self.physical_eeg[index], dtype=np.float32)
        if self.build_graph:
            edge_index, edge_attr = DYNAMIC_GRAPH_BUILDER(source, self.cache_graphs)
        else:
            edge_index = None
            edge_attr = None
        return Data(
            x=torch.from_numpy(source),
            y=torch.from_numpy(target),
            eeg=torch.from_numpy(eeg),
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

def reshape_batch(batch: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    size = batch.num_graphs
    source = batch.x.reshape(size, N_VERTICES, N_TIMES).to(DEVICE)
    target = batch.y.reshape(size, N_VERTICES, N_TIMES).to(DEVICE)
    eeg = batch.eeg.reshape(size, N_CHANNELS, N_TIMES).to(DEVICE)
    return source, target, eeg

def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> tuple[PyGDataLoader, torch.Generator]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = PyGDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )
    return loader, generator
# =============================================================================
# 8. NOISE WHITENING AND COMPOSITE LOSS
# =============================================================================
NOISE_WHITENER_TENSOR: Optional[torch.Tensor] = None

def set_noise_whitener(covariance: np.ndarray) -> None:
    global NOISE_WHITENER_TENSOR
    values, vectors = np.linalg.eigh(np.asarray(covariance, dtype=np.float64))
    floor = max(float(values.max()) * 1e-8, 1e-30)
    inverse_sqrt = vectors @ np.diag(1.0 / np.sqrt(np.maximum(values, floor))) @ vectors.T
    NOISE_WHITENER_TENSOR = torch.tensor(inverse_sqrt, dtype=torch.float32, device=DEVICE)

def composite_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    physical_eeg: torch.Tensor,
    target_scale: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ")

    waveform = F.mse_loss(prediction, target)
    predicted_map = prediction.abs().amax(-1)
    target_map = target.abs().amax(-1)
    target_peak = target_map.amax(1, keepdim=True).clamp_min(1e-8)
    active = target_map >= CFG.active_ratio * target_peak
    weights = 1.0 + (CFG.active_weight - 1.0) * active.float()
    map_loss = (weights * (predicted_map - target_map).square()).sum() / weights.sum().clamp_min(1.0)

    residual = predicted_map - target_map
    residual_smoothness = (
        residual[:, SPATIAL_EDGE_TENSOR[0]] - residual[:, SPATIAL_EDGE_TENSOR[1]]
    ).square().mean()

    predicted_probability = predicted_map / predicted_map.sum(1, keepdim=True).clamp_min(1e-8)
    target_probability = target_map / target_map.sum(1, keepdim=True).clamp_min(1e-8)
    p2t_terms, t2p_terms = [], []
    tau = CFG.geodesic_temperature_mm
    for sample_index in range(len(prediction)):
        true_support = torch.where(active[sample_index])[0]
        if len(true_support) == 0:
            true_support = target_map[sample_index].argmax().reshape(1)
        distance_to_target = GEODESIC_TENSOR[:, true_support].amin(1)
        soft_distance_to_prediction = -tau * torch.logsumexp(
            -GEODESIC_TENSOR / tau + torch.log(predicted_probability[sample_index] + 1e-12)[None, :],
            dim=1,
        )
        p2t_terms.append((predicted_probability[sample_index] * distance_to_target).sum() / 100.0)
        t2p_terms.append((target_probability[sample_index] * soft_distance_to_prediction).sum() / 100.0)
    geodesic = (torch.stack(p2t_terms).mean() + torch.stack(t2p_terms).mean()) / 2.0

    prediction_physical = prediction * target_scale
    reconstructed = torch.einsum("cv,bvt->bct", LEAD_FIELD_TENSOR, prediction_physical)
    residual_eeg = reconstructed - physical_eeg
    denominator_eeg = physical_eeg
    if CFG.whiten_forward_loss and NOISE_WHITENER_TENSOR is not None:
        residual_eeg = torch.einsum("cd,bdt->bct", NOISE_WHITENER_TENSOR, residual_eeg)
        denominator_eeg = torch.einsum("cd,bdt->bct", NOISE_WHITENER_TENSOR, physical_eeg)
    forward = residual_eeg.square().mean() / denominator_eeg.square().mean().clamp_min(1e-12)

    weighted_smoothness = CFG.residual_smoothness_weight * residual_smoothness
    weighted_geodesic = CFG.geodesic_weight * geodesic
    weighted_forward = CFG.forward_weight * forward
    total = waveform + map_loss + weighted_smoothness + weighted_geodesic + weighted_forward

    return total, {
        "total": total.detach(),
        "waveform": waveform.detach(),
        "map": map_loss.detach(),
        "residual_smoothness": residual_smoothness.detach(),
        "geodesic": geodesic.detach(),
        "forward": forward.detach(),
        "weighted_residual_smoothness": weighted_smoothness.detach(),
        "weighted_geodesic": weighted_geodesic.detach(),
        "weighted_forward": weighted_forward.detach(),
    }

# =============================================================================
# 9. TRAINING AND CHECKPOINTS
# =============================================================================
def run_model(model_name: str, model: nn.Module, batch: Data, source: torch.Tensor) -> torch.Tensor:
    return model(batch.to(DEVICE)) if model_name == "physics_gat" else model(source)

def execute_epoch(
    model_name: str,
    model: nn.Module,
    loader: PyGDataLoader,
    optimizer: torch.optim.Optimizer,
    training: bool,
    target_scale: float,
) -> dict[str, float]:
    model.train(training)
    names = (
        "total", "waveform", "map", "residual_smoothness",
        "geodesic", "forward", "weighted_residual_smoothness",
        "weighted_geodesic", "weighted_forward",
    )
    totals = {name: 0.0 for name in names}
    total_samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step, batch in enumerate(loader):
            source, target, eeg = reshape_batch(batch)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = run_model(model_name, model, batch, source)
            loss, components = composite_loss(prediction, target, eeg, target_scale)
            if step % 100 == 0:
                assert_finite("prediction", prediction)
                assert_finite("loss", loss)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            batch_size = source.shape[0]
            total_samples += batch_size
            for name in names:
                totals[name] += float(components[name].cpu()) * batch_size
    return {name: value / max(total_samples, 1) for name, value in totals.items()}

def configuration_hash(seed: int) -> str:
    payload = {
        "schema": SCHEMA_VERSION,
        "config": asdict(CFG),
        "resolved": RUN,
        "seed": seed,
        "lead_field_sha256": sha256_array(lead_field),
        "channels": forward_channels,
        "lh_vertices_sha256": sha256_array(SOURCE_VERTICES[0]),
        "rh_vertices_sha256": sha256_array(SOURCE_VERTICES[1]),
        "environment": environment_manifest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

def train_model(
    model_name: str,
    model: nn.Module,
    train_loader: PyGDataLoader,
    validation_loader: PyGDataLoader,
    train_generator: torch.Generator,
    best_path: Path,
    target_scale: float,
    run_hash: str,
) -> tuple[nn.Module, pd.DataFrame, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=CFG.scheduler_patience, min_lr=1e-6
    )
    last_path = best_path.with_name(best_path.name.replace("best_", "last_"))
    start_epoch, best_validation, stale_epochs = 1, float("inf"), 0
    history: list[dict[str, Any]] = []

    if CFG.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=DEVICE, weights_only=False)
        if checkpoint.get("configuration_hash") == run_hash:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_validation = float(checkpoint["best_validation"])
            stale_epochs = int(checkpoint["stale_epochs"])
            history = checkpoint.get("history", [])
            random.setstate(checkpoint["python_rng"])
            np.random.set_state(checkpoint["numpy_rng"])
            torch.set_rng_state(checkpoint["torch_rng"].cpu())
            train_generator.set_state(checkpoint["loader_rng"].cpu())
            if torch.cuda.is_available() and checkpoint.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
        else:
            warnings.warn(f"Ignored incompatible checkpoint {last_path}")

    started = time.time()
    for epoch in range(start_epoch, RUN["epochs"] + 1):
        train_stats = execute_epoch(model_name, model, train_loader, optimizer, True, target_scale)
        validation_stats = execute_epoch(model_name, model, validation_loader, optimizer, False, target_scale)
        scheduler.step(validation_stats["total"])
        improved = validation_stats["total"] < best_validation - 1e-7
        if improved:
            best_validation, stale_epochs = validation_stats["total"], 0
        else:
            stale_epochs += 1
        row: dict[str, Any] = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{key}": value for key, value in train_stats.items()})
        row.update({f"validation_{key}": value for key, value in validation_stats.items()})
        history.append(row)
        checkpoint = {
            "configuration_hash": run_hash,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_validation": best_validation,
            "stale_epochs": stale_epochs,
            "history": history,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader_rng": train_generator.get_state(),
        }
        torch.save(checkpoint, last_path)
        if improved:
            torch.save(checkpoint, best_path)
        print(model_name, row)
        if stale_epochs >= CFG.early_stopping_patience:
            break

    if not best_path.exists():
        raise RuntimeError(f"No best checkpoint produced for {model_name}")
    best = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(best["model"])
    model.eval()
    return model, pd.DataFrame(history), (time.time() - started) / 60.0

# =============================================================================
# 10. ROBUST METRICS
# =============================================================================
def normalize_map(source_map: np.ndarray) -> np.ndarray:
    source_map = np.asarray(source_map, dtype=np.float64)
    scale = float(np.max(np.abs(source_map))) if source_map.size else 0.0
    return np.zeros_like(source_map) if scale < 1e-12 else source_map / scale

def safe_cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first = normalize_map(first).ravel()
    second = normalize_map(second).ravel()
    norm_first, norm_second = np.linalg.norm(first), np.linalg.norm(second)
    if norm_first < 1e-12 and norm_second < 1e-12:
        return 1.0
    if norm_first < 1e-12 or norm_second < 1e-12:
        return 0.0
    return float(np.dot(first, second) / (norm_first * norm_second))

def safe_spearman(first: np.ndarray, second: np.ndarray) -> tuple[float, bool]:
    first = np.asarray(first).ravel()
    second = np.asarray(second).ravel()
    if np.ptp(first) < 1e-12 or np.ptp(second) < 1e-12:
        return float("nan"), False
    value = float(spearmanr(first, second).statistic)
    return value, bool(np.isfinite(value))

def extract_separated_peaks(activity: np.ndarray) -> np.ndarray:
    activity = np.asarray(activity, dtype=float)
    if activity.size == 0 or np.max(np.abs(activity)) < 1e-12:
        return np.empty(0, dtype=np.int64)
    candidates = np.where(activity >= CFG.peak_ratio * activity.max())[0]
    candidates = candidates[np.argsort(activity[candidates])[::-1]]
    selected: list[int] = []
    for candidate in candidates:
        if not selected or np.all(GEODESIC_MM[candidate, selected] >= CFG.peak_separation_mm):
            selected.append(int(candidate))
        if len(selected) >= CFG.maximum_peaks:
            break
    return np.asarray(selected, dtype=np.int64)

def target_component_centres(target_map: np.ndarray) -> np.ndarray:
    """Derive known-target support components; simulator metadata can replace this."""
    normalized = normalize_map(target_map)
    active = normalized >= CFG.active_ratio
    active_indices = np.where(active)[0]
    if active_indices.size == 0:
        return np.asarray([int(np.argmax(target_map))], dtype=np.int64)
    first, second = UNDIRECTED_EDGES
    keep = active[first] & active[second]
    local_lookup = {int(vertex): i for i, vertex in enumerate(active_indices)}
    rows = np.asarray([local_lookup[int(v)] for v in first[keep]], dtype=np.int64)
    cols = np.asarray([local_lookup[int(v)] for v in second[keep]], dtype=np.int64)
    graph = coo_matrix(
        (np.ones(rows.size * 2), (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
        shape=(active_indices.size, active_indices.size),
    ).tocsr()
    n_components, labels = connected_components(graph, directed=False)
    centres = []
    for component in range(n_components):
        vertices = active_indices[labels == component]
        if len(vertices):
            centres.append(int(vertices[np.argmax(target_map[vertices])]))
    centres = sorted(centres, key=lambda vertex: target_map[vertex], reverse=True)
    return np.asarray(centres[: CFG.maximum_peaks], dtype=np.int64)

def localization_metrics(true_map: np.ndarray, predicted_map: np.ndarray) -> dict[str, float]:
    true_centres = target_component_centres(true_map)
    predicted_centres = extract_separated_peaks(predicted_map)
    if len(predicted_centres) == 0:
        return {
            "dle_mm": CFG.unmatched_penalty_mm,
            "missed_target_components": float(len(true_centres)),
            "extra_predicted_peaks": 0.0,
            "predicted_peaks": 0.0,
        }
    cost = GEODESIC_MM[np.ix_(true_centres, predicted_centres)]
    true_assignment, predicted_assignment = linear_sum_assignment(cost)
    distances = cost[true_assignment, predicted_assignment].tolist()
    missed = len(true_centres) - len(true_assignment)
    extra = len(predicted_centres) - len(predicted_assignment)
    distances.extend([CFG.unmatched_penalty_mm] * (missed + extra))
    return {
        "dle_mm": float(np.mean(distances)),
        "missed_target_components": float(missed),
        "extra_predicted_peaks": float(extra),
        "predicted_peaks": float(len(predicted_centres)),
    }

def sample_metrics(target: np.ndarray, prediction: np.ndarray, leadfield: np.ndarray) -> dict[str, float]:
    target_map = np.max(np.abs(target), axis=-1)
    predicted_map = np.max(np.abs(prediction), axis=-1)
    target_normalized = normalize_map(target_map)
    predicted_normalized = normalize_map(predicted_map)
    target_label = (target_normalized >= CFG.active_ratio).astype(np.int64)
    predicted_label = (predicted_normalized >= CFG.active_ratio).astype(np.int64)
    auc = roc_auc_score(target_label, predicted_map) if np.unique(target_label).size == 2 else np.nan
    precision, recall, f1, _ = precision_recall_fscore_support(
        target_label, predicted_label, average="binary", zero_division=0
    )
    target_sensor = leadfield @ target
    predicted_sensor = leadfield @ prediction
    sensor_residual = predicted_sensor - target_sensor
    metrics = {
        # Physical/calibration-sensitive metrics.
        "physical_waveform_mse": float(np.mean((target - prediction) ** 2)),
        "source_amplitude_ratio": float(np.linalg.norm(prediction) / max(np.linalg.norm(target), 1e-12)),
        "sensor_explained_variance": float(1.0 - np.sum(sensor_residual**2) / max(np.sum(target_sensor**2), 1e-30)),
        # Shape-normalized metrics.
        "normalized_map_mse": float(np.mean((target_normalized - predicted_normalized) ** 2)),
        "cosine": safe_cosine_similarity(target_map, predicted_map),
        "auc": float(auc),
        "relative_threshold_precision": float(precision),
        "relative_threshold_recall": float(recall),
        "relative_threshold_f1": float(f1),
        "relative_extent_error_vertices": float(abs(predicted_label.sum() - target_label.sum())),
    }
    metrics.update(localization_metrics(target_map, predicted_map))
    return metrics
# =============================================================================
# 11. CONVDIP ADAPTER: RECURSIVE FLATTENING, VERTEX CHECK, TIME ALIGNMENT
# =============================================================================
def flatten_predictions(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(flatten_predictions(item))
        return flattened
    return [value]

def expected_vertex_vector() -> np.ndarray:
    return np.concatenate(SOURCE_VERTICES)

def align_source_times(data: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    source_times = np.asarray(source_times, dtype=np.float64)
    if source_times.ndim != 1 or len(source_times) != data.shape[1]:
        raise RuntimeError("ConvDip time metadata does not match output data")
    if source_times[0] > target_times[0] + 1e-3 or source_times[-1] < target_times[-1] - 1e-3:
        raise RuntimeError(
            f"ConvDip output does not cover target interval: "
            f"[{source_times[0]}, {source_times[-1]}] vs [{target_times[0]}, {target_times[-1]}]"
        )
    aligned = np.empty((data.shape[0], len(target_times)), dtype=np.float32)
    for vertex in range(data.shape[0]):
        aligned[vertex] = np.interp(target_times, source_times, data[vertex])
    return aligned

def convdip_item_to_array(item: Any) -> np.ndarray:
    if hasattr(item, "data"):
        array = np.asarray(item.data)
        if hasattr(item, "vertices"):
            returned = np.concatenate([np.asarray(v, dtype=np.int64) for v in item.vertices])
            if not np.array_equal(returned, expected_vertex_vector()):
                raise RuntimeError("ConvDip vertex order differs from fixed forward model")
        if hasattr(item, "times"):
            array = align_source_times(array, np.asarray(item.times), TARGET_TIMES)
        elif array.shape[1] != N_TIMES:
            raise RuntimeError("ConvDip output has no time metadata and incompatible length")
    else:
        array = np.asarray(item)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise RuntimeError(f"Unexpected ConvDip item shape {array.shape}")
    if array.shape[1] != N_TIMES:
        raise RuntimeError("Raw ConvDip array has incompatible time length; refusing crop/pad")
    if array.shape != (N_VERTICES, N_TIMES):
        raise RuntimeError(f"Unexpected ConvDip source shape {array.shape}")
    return array.astype(np.float32, copy=False)

def convdip_output_to_array(prediction: Any) -> np.ndarray:
    items = flatten_predictions(prediction)
    if not items:
        raise RuntimeError("ConvDip returned no predictions")
    return np.stack([convdip_item_to_array(item) for item in items], axis=0)

def predict_convdip_synthetic(model: Net, eeg: np.ndarray) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for sample in eeg:
        evoked = mne.EvokedArray(sample, epochs.info.copy(), tmin=0.0)
        try:
            prediction = model.predict(evoked)
        except Exception as first_error:
            try:
                temporary = Simulation(fwd_fixed, epochs.info.copy())
                temporary.eeg_data = [evoked]
                prediction = model.predict(temporary)
            except Exception as second_error:
                raise RuntimeError("ConvDip synthetic prediction failed") from second_error
        converted = convdip_output_to_array(prediction)
        if len(converted) != 1:
            raise RuntimeError(f"Expected one ConvDip output, received {len(converted)}")
        outputs.append(converted[0])
    return np.stack(outputs, axis=0)

# =============================================================================
# 12. SIMULATION AND SENSOR OOD
# =============================================================================
def make_simulation(number: int, seed: int, snr_db: float, extents: tuple[int, int], sources: int) -> Simulation:
    seed_all(seed)
    simulation = Simulation(
        fwd_fixed,
        epochs.info.copy(),
        settings={
            "duration_of_trial": CFG.duration_s,
            "number_of_sources": sources,
            "extents": extents,
            "target_snr": snr_db,
        },
    )
    simulation.simulate(n_samples=number)
    return simulation

def extract_simulation(simulation: Simulation) -> tuple[np.ndarray, np.ndarray]:
    eeg = extract_collection(simulation.eeg_data)
    source = extract_collection(simulation.source_data)
    if eeg.shape[1:] != (N_CHANNELS, N_TIMES):
        raise RuntimeError(f"Unexpected EEG shape {eeg.shape}")
    if source.shape[1:] != (N_VERTICES, N_TIMES):
        raise RuntimeError(f"Unexpected source shape {source.shape}")
    assert_finite("simulation EEG", eeg)
    assert_finite("simulation source", source)
    return eeg, source

def corrupt_channels(eeg: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = eeg.copy()
    n_drop = max(1, int(round(N_CHANNELS * CFG.ood_channel_dropout_fraction)))
    dropped = rng.choice(N_CHANNELS, size=n_drop, replace=False)
    output[:, dropped, :] = 0.0
    per_sample_sd = np.std(output, axis=(1, 2), keepdims=True)
    noise = rng.normal(size=output.shape).astype(np.float32)
    output += noise * per_sample_sd * CFG.ood_channel_noise_sd_ratio
    return output.astype(np.float32)

# =============================================================================
# 13. TIMING AND STATISTICS
# =============================================================================
def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def benchmark_callable(function: Any) -> dict[str, float]:
    for _ in range(CFG.timing_warmups):
        function()
    synchronize()
    values = []
    for _ in range(CFG.timing_repeats):
        synchronize()
        started = time.perf_counter()
        function()
        synchronize()
        values.append((time.perf_counter() - started) * 1000.0)
    array = np.asarray(values)
    return {
        "median_ms": float(np.median(array)),
        "iqr_ms": float(np.percentile(array, 75) - np.percentile(array, 25)),
        "p95_ms": float(np.percentile(array, 95)),
        "repeats": float(len(array)),
    }

def paired_bootstrap_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty: return pd.DataFrame()
    metric_columns = [
        "physical_waveform_mse",
        "normalized_map_mse",
        "cosine",
        "auc",
        "relative_threshold_f1",
        "dle_mm",
        "sensor_explained_variance",
    ]
    rows = []
    rng = np.random.default_rng(2026)
    algorithms = sorted(frame["algorithm"].unique())
    for condition in sorted(frame["condition"].unique()):
        condition_frame = frame[frame["condition"] == condition]
        for metric in metric_columns:
            pivot = condition_frame.pivot_table(
                index=["seed", "sample"], columns="algorithm", values=metric, aggfunc="first"
            )
            for first_index, first in enumerate(algorithms):
                for second in algorithms[first_index + 1:]:
                    if first not in pivot or second not in pivot: continue
                    differences = (pivot[first] - pivot[second]).dropna().to_numpy()
                    if len(differences) == 0: continue
                    boot = np.empty(CFG.bootstrap_repeats)
                    for index in range(CFG.bootstrap_repeats):
                        sample = rng.choice(differences, size=len(differences), replace=True)
                        boot[index] = sample.mean()
                    rows.append({
                        "condition": condition,
                        "metric": metric,
                        "algorithm_a": first,
                        "algorithm_b": second,
                        "mean_paired_difference_a_minus_b": float(differences.mean()),
                        "ci95_low": float(np.percentile(boot, 2.5)),
                        "ci95_high": float(np.percentile(boot, 97.5)),
                        "paired_samples": int(len(differences)),
                    })
    return pd.DataFrame(rows)

# =============================================================================
# 14. MAIN PIPELINE
# =============================================================================
synthetic_rows: list[dict[str, Any]] = []
real_rows: list[dict[str, Any]] = []
timing_rows: list[dict[str, Any]] = []
forward_check_rows: list[dict[str, Any]] = []
training_rows: list[dict[str, Any]] = []

for seed_index, seed in enumerate(RUN["seeds"]):
    print("=" * 80, "\nSEED", seed, "\n", "=" * 80)
    seed_all(seed)

    train_sim = make_simulation(RUN["n_train"], seed, CFG.train_snr_db, CFG.train_extents, CFG.train_sources)
    validation_sim = make_simulation(RUN["n_validation"], seed + 1, CFG.train_snr_db, CFG.train_extents, CFG.train_sources)
    test_simulations = {
        "ID": make_simulation(RUN["n_test"], seed + 100, CFG.train_snr_db, CFG.train_extents, CFG.train_sources),
        "OOD_SNR": make_simulation(RUN["n_test"], seed + 101, CFG.ood_snr_db, CFG.train_extents, CFG.train_sources),
        "OOD_EXTENT": make_simulation(RUN["n_test"], seed + 102, CFG.train_snr_db, CFG.ood_extents, CFG.train_sources),
        "OOD_SOURCES": make_simulation(RUN["n_test"], seed + 103, CFG.train_snr_db, CFG.train_extents, CFG.ood_sources),
    }

    train_eeg, train_target_raw = extract_simulation(train_sim)
    validation_eeg, validation_target_raw = extract_simulation(validation_sim)
    test_raw = {name: extract_simulation(sim) for name, sim in test_simulations.items()}
    id_eeg, id_source = test_raw["ID"]
    test_raw["OOD_SENSOR_CORRUPTION"] = (corrupt_channels(id_eeg, seed + 104), id_source.copy())

    forward_check_rows.append({
        "seed": seed,
        **validate_forward_consistency(train_eeg[: min(16, len(train_eeg))], train_target_raw[: min(16, len(train_target_raw))], "train_noisy"),
    })

    train_initial_raw = apply_tikhonov(train_eeg)
    validation_initial_raw = apply_tikhonov(validation_eeg)
    input_scale = robust_scale(train_initial_raw)
    target_scale = robust_scale(train_target_raw)

    train_initial_raw /= input_scale
    np.clip(train_initial_raw, -10.0, 10.0, out=train_initial_raw)
    train_initial = maybe_memmap_copy(f"train_initial_{seed}", train_initial_raw)
    del train_initial_raw

    validation_initial_raw /= input_scale
    np.clip(validation_initial_raw, -10.0, 10.0, out=validation_initial_raw)
    validation_initial = validation_initial_raw.astype(np.float32, copy=False)

    train_target = maybe_memmap_copy(
        f"train_target_{seed}", np.clip(train_target_raw / target_scale, -10.0, 10.0).astype(np.float32)
    )
    validation_target = np.clip(validation_target_raw / target_scale, -10.0, 10.0).astype(np.float32)
    test_initial = {
        condition: np.clip(apply_tikhonov(eeg) / input_scale, -10.0, 10.0).astype(np.float32)
        for condition, (eeg, _) in test_raw.items()
    }
    test_target = {
        condition: np.clip(source / target_scale, -10.0, 10.0).astype(np.float32)
        for condition, (_, source) in test_raw.items()
    }

    graph_train, graph_train_gen = make_loader(
        SourceDataset(train_initial, train_target, train_eeg, True, CFG.cache_training_graphs),
        RUN["batch_size"], True, seed + 1_000,
    )
    graph_validation, _ = make_loader(
        SourceDataset(validation_initial, validation_target, validation_eeg, True, True),
        RUN["batch_size"], False, seed + 1_001,
    )
    plain_train, plain_train_gen = make_loader(
        SourceDataset(train_initial, train_target, train_eeg, False, False),
        RUN["batch_size"], True, seed + 2_000,
    )
    plain_validation, _ = make_loader(
        SourceDataset(validation_initial, validation_target, validation_eeg, False, False),
        RUN["batch_size"], False, seed + 2_001,
    )

    # Noise whitener is based only on real pre-stimulus training-side epochs.
    noise_covariance = mne.compute_covariance(
        epochs, tmin=CFG.epoch_tmin, tmax=0.0, method=["shrunk", "empirical"], rank=None, verbose=False,
    )
    covariance_data = np.asarray(noise_covariance.data, dtype=np.float64)
    set_noise_whitener(covariance_data)

    models: dict[str, nn.Module] = {}
    model_offsets = {"tikhonov_cnn": 10, "sparse_graph": 20, "physics_gat": 30}
    for model_name, model_class in {
        "tikhonov_cnn": TikhonovTemporalCNN,
        "sparse_graph": SparseSpatioTemporalGraphNet,
        "physics_gat": FullPhysicsGAT,
    }.items():
        seed_all(seed + model_offsets[model_name])
        models[model_name] = model_class().to(DEVICE)

    if seed_index == 0:
        batch = next(iter(graph_train))
        source, target, eeg = reshape_batch(batch)

        prediction = models["physics_gat"](batch.to(DEVICE))

        smoke_loss, smoke_components = composite_loss(
            prediction,
            target,
            eeg,
            target_scale,
        )

        assert_finite(
            "smoke prediction",
            prediction,
        )
        assert_finite(
            "smoke loss",
            smoke_loss,
        )

        smoke_loss.backward()

        models["physics_gat"].zero_grad(
            set_to_none=True
        )

        print(
            "Forward/backward smoke test passed",
            {
                name: float(component.cpu())
                for name, component
                in smoke_components.items()
            },
        )
    run_hash = configuration_hash(seed)

    trained_models: dict[str, nn.Module] = {}

    for model_name, model in models.items():
        is_gat = model_name == "physics_gat"

        train_loader = (
            graph_train
            if is_gat
            else plain_train
        )

        validation_loader = (
            graph_validation
            if is_gat
            else plain_validation
        )

        generator = (
            graph_train_gen
            if is_gat
            else plain_train_gen
        )

        best_path = (
            OUT
            / f"best_{model_name}seed{seed}.pt"
        )

        trained, history, minutes = train_model(
            model_name,
            model,
            train_loader,
            validation_loader,
            generator,
            best_path,
            target_scale,
            run_hash,
        )

        trained_models[model_name] = trained

        history.to_csv(
            OUT
            / f"history_{model_name}seed{seed}.csv",
            index=False,
        )

        training_rows.append(
            {
                "seed": seed,
                "algorithm": model_name,
                "training_minutes": minutes,
            }
        )

    convdip_model: Optional[Net] = None

    if CFG.train_convdip:
        seed_all(seed + 40)

        convdip_model = Net(fwd_fixed)

        started = time.time()

        convdip_model.fit(
            train_sim,
            epochs=RUN["epochs"],
            batch_size=RUN["batch_size"],
        )

        training_rows.append(
            {
                "seed": seed,
                "algorithm": "convdip",
                "training_minutes": (
                    time.time() - started
                )
                / 60.0,
            }
        )

        try:
            (OUT / f"convdip_seed_{seed}").mkdir(parents=True, exist_ok=True)
            OUT.mkdir(parents=True, exist_ok=True)
            convdip_model.save(str(OUT / f"convdip_seed_{seed}.keras"))
        except Exception as error:
            warnings.warn(
                f"ConvDip save failed: {error}"
            )

    # SYNTHETIC EVALUATION
    for condition, (physical_eeg, target_raw) in test_raw.items():
        initial = test_initial[condition]
        target_normalized = test_target[condition]

        predictions_physical: dict[str, np.ndarray] = {}

        with torch.no_grad():
            source_tensor = (
                torch.from_numpy(initial)
                .to(DEVICE)
            )

            # -------------------------------------------------------------
            # Tikhonov CNN and sparse graph predictions
            # -------------------------------------------------------------

            for algorithm in (
                "tikhonov_cnn",
                "sparse_graph",
            ):
                predicted_normalized = (
                    trained_models[algorithm](source_tensor)
                )

                predictions_physical[
                    algorithm
                ] = (
                    predicted_normalized
                    .cpu()
                    .numpy()
                    * target_scale
                )

            # -------------------------------------------------------------
            # Physics-GAT predictions
            # -------------------------------------------------------------

            graph_dataset = SourceDataset(
                initial_source=initial,
                target_source=target_normalized,
                physical_eeg=physical_eeg,
                build_graph=True,
                cache_graphs=True,
            )

            graph_loader, _ = make_loader(
                graph_dataset,
                batch_size=RUN["batch_size"],
                shuffle=False,
                seed=seed + 3_000,
            )

            graph_predictions: list[
                np.ndarray
            ] = []

            for graph_batch in graph_loader:
                graph_batch = graph_batch.to(
                    DEVICE
                )

                graph_prediction = (
                    trained_models[
                        "physics_gat"
                    ](graph_batch)
                )

                graph_predictions.append(
                    graph_prediction
                    .cpu()
                    .numpy()
                )

            if not graph_predictions:
                raise RuntimeError(
                    "Physics-GAT produced no "
                    f"predictions for {condition}."
                )

            predictions_physical[
                "physics_gat"
            ] = (
                np.concatenate(
                    graph_predictions,
                    axis=0,
                )
                * target_scale
            )

        # -------------------------------------------------------------
        # ConvDip predictions
        # -------------------------------------------------------------
        if convdip_model is not None:
            predictions_physical[
                "convdip"
            ] = predict_convdip_synthetic(
                convdip_model,
                physical_eeg,
            )

        # -------------------------------------------------------------
        # Validate and evaluate all predictions
        # -------------------------------------------------------------
        for (
            algorithm,
            prediction,
        ) in predictions_physical.items():
            if (
                prediction.shape
                != target_raw.shape
            ):
                raise RuntimeError(
                    f"{algorithm}: prediction "
                    f"shape {prediction.shape} "
                    "does not match target "
                    f"shape {target_raw.shape}"
                )

            assert_finite(
                f"{algorithm} prediction",
                prediction,
            )

            for sample_index in range(
                len(target_raw)
            ):
                row = {
                    "seed": seed,
                    "condition": condition,
                    "algorithm": algorithm,
                    "sample": sample_index,
                }

                row.update(
                    sample_metrics(
                        target_raw[
                            sample_index
                        ],
                        prediction[
                            sample_index
                        ],
                        lead_field,
                    )
                )

                synthetic_rows.append(row)

    # =========================================================================
    # REAL EEG AGREEMENT WITH SAME-FORWARD-MODEL dSPM
    # =========================================================================
    if CFG.run_real_eeg:
        inverse_operator = mne.minimum_norm.make_inverse_operator(
            epochs.info,
            fwd_free,
            noise_covariance,
            loose=0.2,
            depth=0.8,
            rank=None,
            verbose=False,
        )

        for condition, evoked in evoked_conditions.items():
            reference = mne.minimum_norm.apply_inverse(
                evoked,
                inverse_operator,
                lambda2=snr_db_to_lambda2(
                    CFG.train_snr_db
                ),
                method="dSPM",
                pick_ori=None,
                verbose=False,
            )

            reference_map = np.max(
                np.abs(reference.data),
                axis=1,
            )

            physical_eeg = np.asarray(
                evoked.data,
                dtype=np.float32,
            )

            if physical_eeg.shape != (
                N_CHANNELS,
                N_TIMES,
            ):
                raise RuntimeError(
                    f"Unexpected real EEG shape for "
                    f"{condition}: {physical_eeg.shape}; "
                    f"expected {(N_CHANNELS, N_TIMES)}"
                )

            assert_finite(
                f"{condition} real EEG",
                physical_eeg,
            )

            # Common Tikhonov preprocessing for custom models.
            initial = np.clip(
                (
                    TIKHONOV_OPERATOR
                    @ physical_eeg
                )
                / input_scale,
                -10.0,
                10.0,
            ).astype(np.float32)

            assert_finite(
                f"{condition} Tikhonov initialization",
                initial,
            )

            source_tensor = torch.from_numpy(
                initial[None]
            ).to(DEVICE)

            predicted_maps: dict[
                str,
                np.ndarray,
            ] = {}

            # -------------------------------------------------------------
            # Tikhonov temporal CNN and sparse graph model
            # -------------------------------------------------------------

            for algorithm in (
                "tikhonov_cnn",
                "sparse_graph",
            ):
                model = trained_models[algorithm]

                with torch.no_grad():
                    predicted = model(
                        source_tensor
                    )

                predicted_maps[algorithm] = (
                    predicted
                    .abs()
                    .amax(-1)[0]
                    .cpu()
                    .numpy()
                    * target_scale
                )

                timing = benchmark_callable(
                    lambda current_model=model: (
                        current_model(
                            source_tensor
                        )
                    )
                )

                timing_rows.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "algorithm": algorithm,
                        "timing_scope":
                            "network_only",
                        **timing,
                    }
                )

            # -------------------------------------------------------------
            # Physics-GAT prediction
            # -------------------------------------------------------------

            edge_index, edge_attr = (
                DYNAMIC_GRAPH_BUILDER(
                    initial,
                    True,
                )
            )

            graph_data = Data(
                x=torch.from_numpy(initial),
                edge_index=edge_index,
                edge_attr=edge_attr,
            )

            graph_loader, _ = make_loader(
                [graph_data],
                batch_size=1,
                shuffle=False,
                seed=seed + 4_000,
            )

            graph_batch_data = next(
                iter(graph_loader)
            ).to(DEVICE)

            with torch.no_grad():
                predicted = trained_models[
                    "physics_gat"
                ](graph_batch_data)

            predicted_maps["physics_gat"] = (
                predicted
                .abs()
                .amax(-1)[0]
                .cpu()
                .numpy()
                * target_scale
            )

            # End-to-end Physics-GAT timing includes:
            # 1. Tikhonov initialization
            # 2. Dynamic graph construction
            # 3. Neural-network inference
            def gat_end_to_end() -> None:
                local_initial = np.clip(
                    (
                        TIKHONOV_OPERATOR
                        @ physical_eeg
                    )
                    / input_scale,
                    -10.0,
                    10.0,
                ).astype(np.float32)

                (
                    local_edges,
                    local_attributes,
                ) = DYNAMIC_GRAPH_BUILDER(
                    local_initial,
                    False,
                )

                local_data = Data(
                    x=torch.from_numpy(
                        local_initial
                    ).to(DEVICE),
                    edge_index=local_edges.to(
                        DEVICE
                    ),
                    edge_attr=local_attributes.to(
                        DEVICE
                    ),
                    batch=torch.zeros(
                        N_VERTICES,
                        dtype=torch.long,
                        device=DEVICE,
                    ),
                )

                with torch.no_grad():
                    trained_models[
                        "physics_gat"
                    ](
                        local_data
                    )

            timing_rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "algorithm": "physics_gat",
                    "timing_scope":
                        "tikhonov_graph_and_network",
                    **benchmark_callable(
                        gat_end_to_end
                    ),
                }
            )

            # -------------------------------------------------------------
            # ConvDip prediction
            # -------------------------------------------------------------

            if convdip_model is not None:
                convdip_prediction = (
                    convdip_model.predict(
                        evoked
                    )
                )

                convdip_array = (
                    convdip_output_to_array(
                        convdip_prediction
                    )
                )

                if len(convdip_array) != 1:
                    raise RuntimeError(
                        "Expected one real-EEG "
                        "ConvDip prediction, but "
                        f"received {len(convdip_array)}"
                    )

                predicted_maps["convdip"] = (
                    np.max(
                        np.abs(
                            convdip_array[0]
                        ),
                        axis=1,
                    )
                )

                timing_rows.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "algorithm": "convdip",
                        "timing_scope":
                            "model_predict",
                        **benchmark_callable(
                            lambda: (
                                convdip_model.predict(
                                    evoked
                                )
                            )
                        ),
                    }
                )

            # -------------------------------------------------------------
            # Real EEG agreement metrics
            # -------------------------------------------------------------

            for (
                algorithm,
                predicted_map,
            ) in predicted_maps.items():
                if (
                    predicted_map.shape
                    != reference_map.shape
                ):
                    raise RuntimeError(
                        f"{algorithm} real-map shape "
                        f"{predicted_map.shape} does "
                        f"not match dSPM shape "
                        f"{reference_map.shape}"
                    )

                assert_finite(
                    (
                        f"{algorithm} real EEG "
                        "predicted map"
                    ),
                    predicted_map,
                )

                reference_normalized = (
                    normalize_map(
                        reference_map
                    )
                )

                predicted_normalized = (
                    normalize_map(
                        predicted_map
                    )
                )

                (
                    spearman,
                    spearman_valid,
                ) = safe_spearman(
                    reference_normalized,
                    predicted_normalized,
                )

                reference_peak = int(
                    np.argmax(
                        reference_map
                    )
                )

                predicted_peak = int(
                    np.argmax(
                        predicted_map
                    )
                )

                peak_distance = (
                    GEODESIC_MM[
                        reference_peak,
                        predicted_peak,
                    ]
                )

                real_rows.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "algorithm": algorithm,
                        "comparator":
                            "dSPM_same_forward_model",
                        "interpretation":
                            "agreement_not_ground_truth",
                        "spearman":
                            spearman,
                        "spearman_valid":
                            spearman_valid,
                        "cosine":
                            safe_cosine_similarity(
                                reference_map,
                                predicted_map,
                            ),
                        "normalized_map_mse":
                            float(
                                np.mean(
                                    (
                                        reference_normalized
                                        - predicted_normalized
                                    )
                                    ** 2
                                )
                            ),
                        "peak_geodesic_mm":
                            float(
                                peak_distance
                            ),
                        "source_amplitude_units":
                            "simulation_calibrated",
                    }
                )

    # =========================================================================
    # RELEASE PER-SEED MEMORY
    # =========================================================================
    del train_sim
    del validation_sim
    del test_simulations

    del train_eeg
    del train_target_raw
    del validation_eeg
    del validation_target_raw

    del train_initial
    del validation_initial
    del train_target
    del validation_target

    del test_raw
    del test_initial
    del test_target

    del trained_models
    del models

    if convdip_model is not None:
        del convdip_model

    DYNAMIC_GRAPH_BUILDER.cache.clear()

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =============================================================================
# 15. OUTPUT FILES
# =============================================================================
synthetic_frame = pd.DataFrame(synthetic_rows)
real_frame = pd.DataFrame(real_rows)
timing_frame = pd.DataFrame(timing_rows)
training_frame = pd.DataFrame(training_rows)
forward_frame = pd.DataFrame(forward_check_rows)

synthetic_frame.to_csv(OUT / "synthetic_sample_metrics.csv", index=False)
real_frame.to_csv(OUT / "real_eeg_agreement.csv", index=False)
timing_frame.to_csv(OUT / "real_eeg_timing.csv", index=False)
training_frame.to_csv(OUT / "training_timing.csv", index=False)
forward_frame.to_csv(OUT / "forward_consistency_checks.csv", index=False)

# =============================================================================
# 16. SEED-LEVEL AND CROSS-SEED STATISTICS
# =============================================================================
if not synthetic_frame.empty:
    metric_columns = [
        column for column in synthetic_frame.columns
        if column not in {"seed", "condition", "algorithm", "sample"}
        and pd.api.types.is_numeric_dtype(synthetic_frame[column])
    ]

    if not metric_columns:
        raise RuntimeError("No numeric synthetic metric columns were produced.")

    seed_summary = (
        synthetic_frame
        .groupby(["seed", "condition", "algorithm"], as_index=False)[metric_columns]
        .mean()
    )

    seed_summary.to_csv(OUT / "synthetic_seed_summary.csv", index=False)

    cross_seed_summary = (
        seed_summary
        .groupby(["condition", "algorithm"])[metric_columns]
        .agg(["mean", "std", "median"])
    )

    cross_seed_summary.columns = ["_".join(column) for column in cross_seed_summary.columns]
    cross_seed_summary.reset_index().to_csv(OUT / "synthetic_cross_seed_summary.csv", index=False)

    bootstrap_frame = paired_bootstrap_summary(synthetic_frame)
    bootstrap_frame.to_csv(OUT / "paired_bootstrap_differences.csv", index=False)
else:
    warnings.warn("No synthetic metric rows were produced.")

# =============================================================================
# 17. MANIFEST
# =============================================================================
manifest = {
    "schema": SCHEMA_VERSION,
    "configuration": asdict(CFG),
    "resolved": RUN,
    "environment": environment_manifest(),
    "forward_model": {
        "lead_field_sha256": sha256_array(lead_field),
        "channels": forward_channels,
        "lh_vertices_sha256": sha256_array(SOURCE_VERTICES[0]),
        "rh_vertices_sha256": sha256_array(SOURCE_VERTICES[1]),
    },
    "scientific_warnings": [
        ("Real EEG comparisons measure agreement with same-forward-model "
         "dSPM, not ground-truth accuracy."),
        ("Simulation source amplitudes calibrate real-EEG output scale; "
         "they are not validated cortical-current units."),
        ("OOD SNR, extent, source-count, and sensor corruption are "
         "robustness tests, not cross-subject physical generalization."),
        ("True physical generalization requires separate forward models, "
         "sensor layouts, waveform and noise families, and subjects."),
        ("Connected target-support components are used when ESINet "
         "does not expose generator-centre metadata."),
        ("ConvDip behavior and serialization depend on the installed ESINet version."),
    ],
}

with open(OUT / "manifest.json", "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, ensure_ascii=False, indent=2, default=str)

print("Pipeline completed. Results saved to:", OUT)
