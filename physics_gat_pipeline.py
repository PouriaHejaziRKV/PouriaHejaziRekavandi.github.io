# -*- coding: utf-8 -*-
"""
PHYSICS-GAT FINAL EXECUTION-READY PIPELINE
==========================================

Execution order:
1. Keep RUN_MODE = "smoke" and execute the entire script.
2. If all sanity checks pass, set RUN_MODE = "pilot".
3. Use RUN_MODE = "full" only in a persistent high-RAM environment.

This page supersedes the previous draft. It intentionally excludes incomplete
Atlas placeholders and does not label a custom CNN as official ConvDip.
"""

from __future__ import annotations

try:
    import IPython
    ipython = IPython.get_ipython()
    if ipython is not None:
        ipython.system('pip install -q esinet "mne>=1.7,<1.10" pandas seaborn matplotlib scikit-learn scipy torch torch-geometric psutil gdown')
except Exception:
    pass

import gc
import glob
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import random
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

# IMPORTANT: torch and torch_geometric must be imported before esinet/tensorflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import add_self_loops, coalesce

import mne
import numpy as np
import pandas as pd
import psutil
from esinet import Net, Simulation
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

warnings.filterwarnings("default")
mne.set_log_level("WARNING")

try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    pass


@dataclass
class Config:
    run_mode: str = "smoke"  # smoke | pilot | full
    seeds: tuple[int, ...] = (42, 52, 62, 72, 82)

    # Dynamically select paths depending on if we are in Colab with mounted drive
    if os.path.exists("/content/drive/MyDrive"):
        mne_root: str = "/content/drive/MyDrive/Esinet/mne_data"
        output_dir: str = "/content/drive/MyDrive/Esinet/physics_gat_final"
    else:
        mne_root: str = mne.datasets.sample.data_path()
        output_dir: str = "./physics_gat_final"

    source_spacing: str = "ico3"
    sfreq: float = 100.0
    duration_s: float = 0.30
    mindist_mm: float = 5.0

    train_snr: float = 3.0
    ood_snr: float = -10.0
    train_extents: tuple[int, int] = (10, 20)
    ood_extents: tuple[int, int] = (28, 42)
    train_sources: int = 3
    ood_sources: int = 6

    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    scheduler_patience: int = 7
    early_stopping: int = 20

    tikhonov_relative: float = 1e-2
    hidden: int = 32
    heads: int = 4
    dropout: float = 0.15
    nonlocal_k: int = 4
    functional_k: int = 3

    active_ratio: float = 0.10
    active_weight: float = 50.0
    smoothness_weight: float = 1e-4
    geodesic_weight: float = 2e-3
    forward_weight: float = 1e-2
    geodesic_temperature_mm: float = 10.0

    peak_ratio: float = 0.25
    peak_separation_mm: float = 20.0
    maximum_peaks: int = 12
    unmatched_penalty_mm: float = 100.0

    epoch_tmin: float = -0.20
    epoch_tmax: float = 0.50
    inference_tmin: float = 0.00
    inference_tmax: float = 0.30

    train_convdip: bool = True
    run_real_eeg: bool = True
    resume: bool = True

    def run_values(self) -> dict[str, Any]:
        if self.run_mode == "smoke":
            return dict(seeds=(self.seeds[0],), n_train=32, n_validation=8,
                        n_test=8, epochs=1, batch_size=2)
        if self.run_mode == "pilot":
            return dict(seeds=(self.seeds[0],), n_train=1000, n_validation=200,
                        n_test=200, epochs=5, batch_size=8)
        if self.run_mode == "full":
            return dict(seeds=self.seeds, n_train=40000, n_validation=1000,
                        n_test=1000, epochs=150, batch_size=64)
        raise ValueError("run_mode must be smoke, pilot, or full")


CFG = Config()
RUN = CFG.run_values()
OUT = Path(CFG.output_dir)
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if CFG.hidden % CFG.heads != 0:
    raise ValueError("hidden must be divisible by heads")


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
        value = np.asarray(value)
        valid = bool(np.isfinite(value).all())
        shape = value.shape
    if not valid:
        raise RuntimeError(f"{name} contains non-finite values; shape={shape}")


def find_all(patterns: list[str]) -> list[str]:
    results: list[str] = []
    for pattern in patterns:
        results.extend(glob.glob(pattern, recursive=True))
    return sorted(set(results))


def find_first(patterns: list[str]) -> Optional[str]:
    results = find_all(patterns)
    return results[0] if results else None


def find_subjects_dir(root: str) -> Optional[str]:
    candidates = find_all([os.path.join(root, "**", "subjects")]) + [root]
    for directory in candidates:
        if os.path.isdir(os.path.join(directory, "sample", "surf")) and \
           os.path.isdir(os.path.join(directory, "sample", "mri")):
            return directory
    return None


seed_all(RUN["seeds"][0])

# Update mne_root dynamically based on available files if the path is not direct
if not os.path.exists(CFG.mne_root):
    # Try downloading or resolving
    CFG.mne_root = str(mne.datasets.sample.data_path())

raw_file = find_first([
    os.path.join(CFG.mne_root, "**", "sample_audvis_filt-0-40_raw.fif"),
    os.path.join(CFG.mne_root, "**", "sample_audvis_raw.fif"),
])
event_file = find_first([
    os.path.join(CFG.mne_root, "**", "sample_audvis_raw-eve.fif"),
    os.path.join(CFG.mne_root, "**", "*audvis*-eve.fif"),
])
trans_file = find_first([
    os.path.join(CFG.mne_root, "**", "sample_audvis_raw-trans.fif"),
])
bem_file = find_first([
    os.path.join(CFG.mne_root, "**", "sample-5120-5120-5120-bem-sol.fif"),
    os.path.join(CFG.mne_root, "**", "*-bem-sol.fif"),
])
subjects_dir = find_subjects_dir(CFG.mne_root)

for name, value in dict(raw_file=raw_file, event_file=event_file,
                        trans_file=trans_file, bem_file=bem_file,
                        subjects_dir=subjects_dir).items():
    if value is None:
        raise FileNotFoundError(f"{name} was not found under {CFG.mne_root}")

os.environ["SUBJECTS_DIR"] = str(subjects_dir)
mne.set_config("SUBJECTS_DIR", str(subjects_dir), set_env=True)

raw_full = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)
raw = raw_full.copy().pick(picks=["eeg", "eog", "stim"], exclude="bads")
raw.apply_proj()
raw.filter(1.0, 40.0, picks="eeg", method="fir", phase="zero", verbose=False)
raw.set_eeg_reference("average", projection=True, verbose=False)
raw.apply_proj()

try:
    events = mne.find_events(raw_full, stim_channel="STI 014", shortest_event=1,
                             verbose=False)
except Exception:
    events = mne.read_events(event_file)
    events = events[(events[:, 0] >= raw.first_samp) &
                    (events[:, 0] <= raw.last_samp)]

requested_events = {"Auditory_Left": 1, "Auditory_Right": 2,
                    "Visual_Left": 3, "Visual_Right": 4}
event_id = {name: code for name, code in requested_events.items()
            if np.any(events[:, 2] == code)}
if not event_id:
    raise RuntimeError("No requested events were found")

eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False,
                           stim=False, exclude="bads")
epochs = mne.Epochs(
    raw, events, event_id=event_id, tmin=CFG.epoch_tmin, tmax=CFG.epoch_tmax,
    baseline=(CFG.epoch_tmin, 0.0), picks=eeg_picks, reject={"eeg": 150e-6},
    preload=True, proj=True, detrend=1, reject_by_annotation=True,
    on_missing="warn", verbose=False,
)
epochs.resample(CFG.sfreq, npad="auto", verbose=False)

evokeds = {name: epochs[name].average().crop(CFG.inference_tmin,
                                              CFG.inference_tmax)
           for name in event_id if len(epochs[name]) > 0}

del raw_full
gc.collect()

source_space = mne.setup_source_space(
    "sample", spacing=CFG.source_spacing, subjects_dir=str(subjects_dir),
    add_dist=False, verbose=False,
)
fwd_free = mne.make_forward_solution(
    epochs.info, trans=str(trans_file), src=source_space, bem=str(bem_file),
    meg=False, eeg=True, mindist=CFG.mindist_mm, n_jobs=1, verbose=False,
)
fwd_free = mne.pick_types_forward(fwd_free, meg=False, eeg=True,
                                  ref_meg=False, exclude="bads")
common_channels = [ch for ch in epochs.ch_names
                   if ch in fwd_free["info"]["ch_names"]]
if not common_channels:
    raise RuntimeError("No common EEG channels between epochs and forward model")
epochs.pick(common_channels)
for evoked in evokeds.values():
    evoked.pick(common_channels)
fwd_free = mne.pick_channels_forward(fwd_free, include=common_channels,
                                     ordered=True, copy=True)
channel_order = list(fwd_free["info"]["ch_names"])
epochs.reorder_channels(channel_order)
for evoked in evokeds.values():
    evoked.reorder_channels(channel_order)

fwd_fixed = mne.convert_forward_solution(
    fwd_free, surf_ori=True, force_fixed=True, use_cps=True,
    copy=True, verbose=False,
)
LEAD_FIELD = np.asarray(fwd_fixed["sol"]["data"], dtype=np.float32)
N_CHANNELS, N_VERTICES = LEAD_FIELD.shape
N_TIMES = int(round(CFG.duration_s * CFG.sfreq))
COORDINATES_MM = np.vstack([
    src["rr"][src["vertno"]] for src in fwd_fixed["src"]
]).astype(np.float32) * 1000.0

if list(fwd_fixed["sol"]["row_names"]) != channel_order:
    raise RuntimeError("Forward and EEG channel orders are inconsistent")


def cortical_edges() -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    offset = 0
    for hemi in fwd_fixed["src"]:
        used = np.asarray(hemi["vertno"], dtype=np.int64)
        lookup = {int(v): i for i, v in enumerate(used)}
        triangles = hemi.get("use_tris", None)
        if triangles is None:
            triangles = hemi.get("tris", None)
        if triangles is None:
            raise RuntimeError("No cortical triangles found")
        for tri in np.asarray(triangles, dtype=np.int64):
            if not all(int(v) in lookup for v in tri):
                continue
            nodes = [lookup[int(v)] + offset for v in tri]
            for first, second in ((nodes[0], nodes[1]), (nodes[1], nodes[2]),
                                  (nodes[2], nodes[0])):
                if first != second:
                    edges.add(tuple(sorted((first, second))))
        offset += len(used)
    if not edges:
        raise RuntimeError("Cortical graph is empty")
    return np.asarray(sorted(edges), dtype=np.int64).T


UNDIRECTED_EDGES = cortical_edges()
first, second = UNDIRECTED_EDGES
lengths = np.linalg.norm(COORDINATES_MM[first] - COORDINATES_MM[second], axis=1)
mesh_graph = coo_matrix(
    (np.concatenate([lengths, lengths]),
     (np.concatenate([first, second]), np.concatenate([second, first]))),
    shape=(N_VERTICES, N_VERTICES),
).tocsr()
GEODESIC_MM = np.asarray(dijkstra(mesh_graph, directed=False), dtype=np.float32)
finite = GEODESIC_MM[np.isfinite(GEODESIC_MM)]
if finite.size == 0:
    raise RuntimeError("No finite geodesic distances")
GEODESIC_MM[~np.isfinite(GEODESIC_MM)] = finite.max() + 100.0

local_edges = torch.tensor(
    np.vstack([np.concatenate([first, second]), np.concatenate([second, first])]),
    dtype=torch.long,
)
local_edges, _ = add_self_loops(local_edges, num_nodes=N_VERTICES)
LOCAL_EDGE_INDEX = coalesce(local_edges, num_nodes=N_VERTICES)

profiles = LEAD_FIELD.T.copy()
profiles -= profiles.mean(axis=1, keepdims=True)
profiles /= np.linalg.norm(profiles, axis=1, keepdims=True) + 1e-8
lead_similarity = np.abs(profiles @ profiles.T)
selection = lead_similarity.copy()
np.fill_diagonal(selection, -np.inf)
nonlocal_neighbours = np.argpartition(selection, -CFG.nonlocal_k, axis=1)[:,
                                      -CFG.nonlocal_k:]
nonlocal_source = np.repeat(np.arange(N_VERTICES), CFG.nonlocal_k)
nonlocal_target = nonlocal_neighbours.reshape(-1)
nonlocal_edges = torch.tensor(np.vstack([nonlocal_source, nonlocal_target]),
                              dtype=torch.long)
BASE_EDGES = coalesce(torch.cat([LOCAL_EDGE_INDEX, nonlocal_edges,
                                 nonlocal_edges.flip(0)], dim=1),
                      num_nodes=N_VERTICES)


def edge_attributes(edge_index: torch.Tensor,
                    functional: Optional[np.ndarray] = None) -> torch.Tensor:
    first_np = edge_index[0].cpu().numpy()
    second_np = edge_index[1].cpu().numpy()
    euclidean = np.linalg.norm(COORDINATES_MM[first_np] -
                               COORDINATES_MM[second_np], axis=1) / 200.0
    geodesic = GEODESIC_MM[first_np, second_np] / 300.0
    lead = np.abs(np.sum(profiles[first_np] * profiles[second_np], axis=1))
    functional_values = (np.zeros(len(first_np), dtype=np.float32)
                         if functional is None
                         else functional[first_np, second_np].astype(np.float32))
    self_mask = first_np == second_np
    euclidean[self_mask] = 0.0
    geodesic[self_mask] = 0.0
    lead[self_mask] = 1.0
    functional_values[self_mask] = 0.0
    values = np.column_stack([euclidean, geodesic, lead,
                              functional_values]).astype(np.float32)
    assert_finite("edge attributes", values)
    if values.shape != (edge_index.shape[1], 4):
        raise RuntimeError("edge_attr does not match edge_index")
    return torch.from_numpy(values)


class DynamicGraphBuilder:
    def __call__(self, source: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        centered = source - source.mean(axis=1, keepdims=True)
        centered /= np.linalg.norm(centered, axis=1, keepdims=True) + 1e-8
        functional = np.abs(centered @ centered.T).astype(np.float32)
        functional_selection = functional.copy()
        np.fill_diagonal(functional_selection, -np.inf)
        neighbours = np.argpartition(functional_selection, -CFG.functional_k,
                                     axis=1)[:, -CFG.functional_k:]
        source_nodes = np.repeat(np.arange(N_VERTICES), CFG.functional_k)
        target_nodes = neighbours.reshape(-1)
        functional_edges = torch.tensor(np.vstack([source_nodes, target_nodes]),
                                        dtype=torch.long)
        final_edges = coalesce(torch.cat([BASE_EDGES, functional_edges,
                                          functional_edges.flip(0)], dim=1),
                               num_nodes=N_VERTICES)
        np.fill_diagonal(functional, 0.0)
        return final_edges, edge_attributes(final_edges, functional)


GRAPH_BUILDER = DynamicGraphBuilder()


def tikhonov_operator(matrix: np.ndarray) -> np.ndarray:
    matrix64 = matrix.astype(np.float64)
    gram = matrix64 @ matrix64.T
    regularization = CFG.tikhonov_relative * max(np.trace(gram) / gram.shape[0],
                                                 1e-30)
    return np.linalg.solve(gram + regularization * np.eye(gram.shape[0]),
                           matrix64).T.astype(np.float32)


TIKHONOV_OPERATOR = tikhonov_operator(LEAD_FIELD)


def extract_collection(data: Any) -> np.ndarray:
    if isinstance(data, np.ndarray):
        if data.ndim == 3:
            return data.astype(np.float32, copy=False)
        if data.ndim == 2:
            return data[None].astype(np.float32, copy=False)
        raise ValueError(f"Unexpected array shape: {data.shape}")
    arrays = []
    for item in list(data):
        if hasattr(item, "get_data"):
            array = np.asarray(item.get_data())
        elif hasattr(item, "data"):
            array = np.asarray(item.data)
        else:
            array = np.asarray(item)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2:
            raise ValueError(f"Expected a 2D sample; received {array.shape}")
        arrays.append(array.astype(np.float32, copy=False))
    return np.stack(arrays, axis=0)


def make_simulation(n_samples: int, seed: int, snr: float,
                    extents: tuple[int, int], sources: int) -> Simulation:
    seed_all(seed)
    simulation = Simulation(
        fwd_fixed,
        epochs.info.copy(),
        settings={"duration_of_trial": CFG.duration_s,
                  "number_of_sources": sources,
                  "extents": extents,
                  "target_snr": snr},
    )
    simulation.simulate(n_samples=n_samples)
    return simulation


def extract_simulation(sim: Simulation) -> tuple[np.ndarray, np.ndarray]:
    eeg = extract_collection(sim.eeg_data)
    source = extract_collection(sim.source_data)
    if eeg.shape[1:] != (N_CHANNELS, N_TIMES):
        raise RuntimeError(f"Unexpected EEG shape: {eeg.shape}")
    if source.shape[1:] != (N_VERTICES, N_TIMES):
        raise RuntimeError(f"Unexpected source shape: {source.shape}")
    assert_finite("simulation EEG", eeg)
    assert_finite("simulation sources", source)
    return eeg, source


class SourceDataset(Dataset):
    def __init__(self, initial: np.ndarray, target: np.ndarray,
                 physical_eeg: np.ndarray, graph: bool):
        self.initial = initial
        self.target = target
        self.physical_eeg = physical_eeg
        self.graph = graph

    def __len__(self) -> int:
        return len(self.initial)

    def __getitem__(self, index: int) -> Data:
        edge_index, edge_attr = (GRAPH_BUILDER(self.initial[index])
                                 if self.graph else (None, None))
        return Data(x=torch.from_numpy(self.initial[index]),
                    y=torch.from_numpy(self.target[index]),
                    eeg=torch.from_numpy(self.physical_eeg[index]),
                    edge_index=edge_index, edge_attr=edge_attr)


class TemporalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 8, 5, padding=2), nn.GroupNorm(2, 8), nn.GELU(),
            nn.Conv1d(8, 12, 3, padding=1), nn.GroupNorm(3, 12), nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.projection = nn.Linear(48, CFG.hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.network(x[:, None]).flatten(1))


class TikhonovTemporalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TemporalEncoder()
        self.decoder = nn.Linear(CFG.hidden, N_TIMES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, vertices, times = x.shape
        hidden = self.encoder(x.reshape(batch * vertices, times))
        return self.decoder(F.gelu(hidden)).reshape(batch, vertices, N_TIMES)


class SparseGraphNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TemporalEncoder()
        self.proj1 = nn.Linear(CFG.hidden, CFG.hidden)
        self.proj2 = nn.Linear(CFG.hidden, CFG.hidden)
        self.norm1 = nn.LayerNorm(CFG.hidden)
        self.norm2 = nn.LayerNorm(CFG.hidden)
        self.decoder = nn.Linear(CFG.hidden, N_TIMES)
        self.adjacency = self._make_adjacency()

    @staticmethod
    def _make_adjacency() -> torch.Tensor:
        first_np, second_np = UNDIRECTED_EDGES
        self_nodes = np.arange(N_VERTICES)
        rows = np.concatenate([first_np, second_np, self_nodes])
        cols = np.concatenate([second_np, first_np, self_nodes])
        degree = np.maximum(np.bincount(rows, minlength=N_VERTICES), 1)
        values = (1.0 / np.sqrt(degree[rows] * degree[cols])).astype(np.float32)
        return torch.sparse_coo_tensor(
            torch.tensor(np.vstack([rows, cols]), dtype=torch.long,
                         device=DEVICE),
            torch.tensor(values, dtype=torch.float32, device=DEVICE),
            (N_VERTICES, N_VERTICES), device=DEVICE,
        ).coalesce()

    def propagate(self, x: torch.Tensor) -> torch.Tensor:
        batch, vertices, channels = x.shape
        matrix = x.permute(1, 0, 2).reshape(vertices, batch * channels)
        propagated = torch.sparse.mm(self.adjacency, matrix)
        return propagated.reshape(vertices, batch, channels).permute(1, 0, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, vertices, times = x.shape
        hidden = self.encoder(x.reshape(batch * vertices, times)).reshape(batch, vertices, -1)
        hidden = self.norm1(hidden + self.proj1(self.propagate(hidden)))
        hidden = self.norm2(hidden + self.proj2(self.propagate(F.gelu(hidden))))
        return self.decoder(F.gelu(hidden))


class GATBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat = GATv2Conv(CFG.hidden, CFG.hidden // CFG.heads,
                             heads=CFG.heads, edge_dim=4,
                             dropout=CFG.dropout, add_self_loops=False)
        self.norm = nn.LayerNorm(CFG.hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        return self.norm(x + F.dropout(F.gelu(self.gat(x, edge_index, edge_attr)),
                                       CFG.dropout, self.training))


class PhysicsGAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TemporalEncoder()
        self.block1 = GATBlock()
        self.block2 = GATBlock()
        self.decoder = nn.Linear(CFG.hidden, N_TIMES)

    def forward(self, batch: Data) -> torch.Tensor:
        counts = torch.bincount(batch.batch)
        if not torch.all(counts == N_VERTICES):
            raise RuntimeError("PhysicsGAT requires equal full-size graphs")
        hidden = self.encoder(batch.x)
        hidden = self.block1(hidden, batch.edge_index, batch.edge_attr)
        hidden = self.block2(hidden, batch.edge_index, batch.edge_attr)
        return self.decoder(hidden).reshape(batch.num_graphs, N_VERTICES, N_TIMES)


SPATIAL_EDGES = torch.tensor(
    np.vstack([np.concatenate([UNDIRECTED_EDGES[0], UNDIRECTED_EDGES[1]]),
               np.concatenate([UNDIRECTED_EDGES[1], UNDIRECTED_EDGES[0]])]),
    dtype=torch.long, device=DEVICE,
)
GEODESIC_TENSOR = torch.tensor(GEODESIC_MM, dtype=torch.float32, device=DEVICE)
LEAD_FIELD_TENSOR = torch.tensor(LEAD_FIELD, dtype=torch.float32, device=DEVICE)


def composite_loss(prediction: torch.Tensor, target: torch.Tensor,
                   physical_eeg: torch.Tensor,
                   target_scale: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    waveform = F.mse_loss(prediction, target)
    pred_map = prediction.abs().amax(-1)
    target_map = target.abs().amax(-1)
    active = target_map >= CFG.active_ratio * target_map.amax(1, keepdim=True).clamp_min(1e-8)
    weights = 1.0 + (CFG.active_weight - 1.0) * active.float()
    map_loss = (weights * (pred_map - target_map).square()).sum() / weights.sum().clamp_min(1.0)
    residual = pred_map - target_map
    smoothness = (residual[:, SPATIAL_EDGES[0]] -
                  residual[:, SPATIAL_EDGES[1]]).square().mean()

    pred_prob = pred_map / pred_map.sum(1, keepdim=True).clamp_min(1e-8)
    target_prob = target_map / target_map.sum(1, keepdim=True).clamp_min(1e-8)
    p2t, t2p = [], []
    tau = CFG.geodesic_temperature_mm
    for index in range(len(prediction)):
        true_support = torch.where(active[index])[0]
        if len(true_support) == 0:
            true_support = target_map[index].argmax().reshape(1)
        distance_to_target = GEODESIC_TENSOR[:, true_support].amin(1)
        soft_distance = -tau * torch.logsumexp(
            -GEODESIC_TENSOR / tau + torch.log(pred_prob[index] + 1e-12)[None],
            dim=1,
        )
        p2t.append((pred_prob[index] * distance_to_target).sum() / 100.0)
        t2p.append((target_prob[index] * soft_distance).sum() / 100.0)
    geodesic = (torch.stack(p2t).mean() + torch.stack(t2p).mean()) / 2.0

    predicted_eeg = torch.einsum("cv,bvt->bct", LEAD_FIELD_TENSOR,
                                 prediction * target_scale)
    forward = ((predicted_eeg - physical_eeg).square().mean() /
               physical_eeg.square().mean().clamp_min(1e-12))
    total = (waveform + map_loss + CFG.smoothness_weight * smoothness +
             CFG.geodesic_weight * geodesic + CFG.forward_weight * forward)
    components = {"total": total.detach(), "waveform": waveform.detach(),
                  "map": map_loss.detach(), "smoothness": smoothness.detach(),
                  "geodesic": geodesic.detach(), "forward": forward.detach()}
    return total, components


def reshape_batch(batch: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = batch.num_graphs
    return (
        batch.x.reshape(count, N_VERTICES, N_TIMES).to(DEVICE),
        batch.y.reshape(count, N_VERTICES, N_TIMES).to(DEVICE),
        batch.eeg.reshape(count, N_CHANNELS, N_TIMES).to(DEVICE),
    )


def run_model(name: str, model: nn.Module, batch: Data,
              initial: torch.Tensor) -> torch.Tensor:
    if name == "physics_gat":
        return model(batch.to(DEVICE))
    return model(initial)


def execute_epoch(name: str, model: nn.Module, loader: PyGDataLoader,
                  optimizer: torch.optim.Optimizer, training: bool,
                  target_scale: float) -> dict[str, float]:
    model.train(training)
    totals = {key: 0.0 for key in ("total", "waveform", "map", "smoothness",
                                         "geodesic", "forward")}
    samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step, batch in enumerate(loader):
            initial, target, eeg = reshape_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            prediction = run_model(name, model, batch, initial)
            loss, components = composite_loss(prediction, target, eeg, target_scale)
            if step % 100 == 0:
                assert_finite("prediction", prediction)
                assert_finite("loss", loss)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            batch_size = initial.shape[0]
            samples += batch_size
            for key in totals:
                totals[key] += float(components[key].cpu()) * batch_size
    return {key: value / max(samples, 1) for key, value in totals.items()}


def run_hash(seed: int) -> str:
    payload = {"config": asdict(CFG), "run": RUN, "seed": seed,
               "channels": N_CHANNELS, "vertices": N_VERTICES,
               "times": N_TIMES}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str)
                          .encode("utf-8")).hexdigest()


def train_model(name: str, model: nn.Module, train_loader: PyGDataLoader,
                validation_loader: PyGDataLoader, best_path: Path,
                target_scale: float, configuration_hash: str):
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate,
                                  weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=CFG.scheduler_patience, min_lr=1e-6)
    last_path = best_path.with_name(best_path.name.replace("best_", "last_"))
    start_epoch, stale, best = 1, 0, float("inf")
    history = []

    if CFG.resume and last_path.exists():
        try:
            checkpoint = torch.load(last_path, map_location=DEVICE,
                                    weights_only=False)
            if checkpoint.get("hash") != configuration_hash:
                warnings.warn(f"Incompatible checkpoint hash found in {last_path}. Starting fresh.")
            else:
                model.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                scheduler.load_state_dict(checkpoint["scheduler"])
                start_epoch = checkpoint["epoch"] + 1
                stale, best = checkpoint["stale"], checkpoint["best"]
                history = checkpoint.get("history", [])
                random.setstate(checkpoint["python_rng"])
                np.random.set_state(checkpoint["numpy_rng"])
                torch.set_rng_state(checkpoint["torch_rng"])
                if torch.cuda.is_available() and checkpoint.get("cuda_rng") is not None:
                    torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        except Exception as e:
            warnings.warn(f"Failed to load checkpoint {last_path}: {e}. Starting fresh.")

    started = time.time()
    for epoch in range(start_epoch, RUN["epochs"] + 1):
        train_stats = execute_epoch(name, model, train_loader, optimizer, True,
                                    target_scale)
        validation_stats = execute_epoch(name, model, validation_loader,
                                         optimizer, False, target_scale)
        scheduler.step(validation_stats["total"])
        improved = validation_stats["total"] < best - 1e-7
        if improved:
            best, stale = validation_stats["total"], 0
        else:
            stale += 1
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{k}": v for k, v in train_stats.items()})
        row.update({f"validation_{k}": v for k, v in validation_stats.items()})
        history.append(row)
        checkpoint = {
            "hash": configuration_hash, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "stale": stale, "best": best, "history": history,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(checkpoint, last_path)
        if improved:
            torch.save(checkpoint, best_path)
        print(name, row)
        if stale >= CFG.early_stopping:
            break

    if not best_path.exists():
        # Fallback if no best model was explicitly saved during this run
        if last_path.exists():
            torch.save(torch.load(last_path, map_location=DEVICE, weights_only=False), best_path)
        else:
            raise RuntimeError(f"No best checkpoint for {name}")
    checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, pd.DataFrame(history), (time.time() - started) / 60.0


def normalize_map(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return value / (np.max(np.abs(value)) + 1e-12)


def separated_peaks(activity: np.ndarray) -> np.ndarray:
    candidates = np.where(activity >= CFG.peak_ratio * (activity.max() + 1e-12))[0]
    candidates = candidates[np.argsort(activity[candidates])[::-1]]
    selected = []
    for candidate in candidates:
        if not selected or np.all(GEODESIC_MM[candidate, selected] >= CFG.peak_separation_mm):
            selected.append(int(candidate))
        if len(selected) >= CFG.maximum_peaks:
            break
    return np.asarray(selected or [int(np.argmax(activity))], dtype=np.int64)


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target_map = np.max(np.abs(target), axis=-1)
    prediction_map = np.max(np.abs(prediction), axis=-1)
    target_norm, prediction_norm = normalize_map(target_map), normalize_map(prediction_map)
    target_label = (target_norm >= CFG.active_ratio).astype(np.int64)
    prediction_label = (prediction_norm >= CFG.active_ratio).astype(np.int64)
    auc = roc_auc_score(target_label, prediction_map) if np.unique(target_label).size == 2 else np.nan
    precision, recall, f1, _ = precision_recall_fscore_support(
        target_label, prediction_label, average="binary", zero_division=0)
    true_peaks, predicted_peaks = separated_peaks(target_map), separated_peaks(prediction_map)
    cost = GEODESIC_MM[np.ix_(true_peaks, predicted_peaks)]
    true_assignment, prediction_assignment = linear_sum_assignment(cost)
    distances = cost[true_assignment, prediction_assignment].tolist()
    missed = len(true_peaks) - len(true_assignment)
    false = len(predicted_peaks) - len(prediction_assignment)
    distances.extend([CFG.unmatched_penalty_mm] * (missed + false))
    return {
        "waveform_mse": float(np.mean((target - prediction) ** 2)),
        "map_mse": float(np.mean((target_norm - prediction_norm) ** 2)),
        "cosine": float(1.0 - cosine(target_norm, prediction_norm)),
        "auc": float(auc), "precision": float(precision),
        "recall": float(recall), "f1": float(f1),
        "extent_error_vertices": float(abs(prediction_label.sum() - target_label.sum())),
        "dle_mm": float(np.mean(distances)), "missed_sources": float(missed),
        "false_sources": float(false), "predicted_sources": float(len(predicted_peaks)),
    }


def convdip_to_array(prediction: Any) -> np.ndarray:
    if hasattr(prediction, "data"):
        array = np.asarray(prediction.data)[None]
    elif isinstance(prediction, (list, tuple)):
        array = np.stack([np.asarray(item.data) if hasattr(item, "data")
                          else np.asarray(item) for item in prediction])
    else:
        array = np.asarray(prediction)
        if array.ndim == 2:
            array = array[None]
    if array.shape[1:] != (N_VERTICES, N_TIMES):
        raise RuntimeError(f"Unexpected ConvDip output: {array.shape}")
    return array.astype(np.float32, copy=False)


def convdip_synthetic(model: Net, eeg: np.ndarray) -> np.ndarray:
    evoked_list = [mne.EvokedArray(sample, epochs.info.copy(), tmin=0.0)
                   for sample in eeg]
    try:
        prediction = model.predict(evoked_list)
    except Exception:
        temporary = Simulation(fwd_fixed, epochs.info.copy())
        temporary.eeg_data = evoked_list
        prediction = model.predict(temporary)
    return convdip_to_array(prediction)


synthetic_rows, real_rows, timing_rows = [], [], []

for seed_index, seed in enumerate(RUN["seeds"]):
    seed_all(seed)
    print("=" * 80, "\nSEED", seed, "\n", "=" * 80)

    train_sim = make_simulation(RUN["n_train"], seed, CFG.train_snr,
                                CFG.train_extents, CFG.train_sources)
    validation_sim = make_simulation(RUN["n_validation"], seed + 1,
                                     CFG.train_snr, CFG.train_extents,
                                     CFG.train_sources)
    test_sims = {
        "ID": make_simulation(RUN["n_test"], seed + 100, CFG.train_snr,
                              CFG.train_extents, CFG.train_sources),
        "OOD_SNR": make_simulation(RUN["n_test"], seed + 101, CFG.ood_snr,
                                   CFG.train_extents, CFG.train_sources),
        "OOD_EXTENT": make_simulation(RUN["n_test"], seed + 102,
                                      CFG.train_snr, CFG.ood_extents,
                                      CFG.train_sources),
        "OOD_SOURCES": make_simulation(RUN["n_test"], seed + 103,
                                       CFG.train_snr, CFG.train_extents,
                                       CFG.ood_sources),
    }

    train_eeg, train_target_raw = extract_simulation(train_sim)
    validation_eeg, validation_target_raw = extract_simulation(validation_sim)
    test_raw = {name: extract_simulation(sim) for name, sim in test_sims.items()}

    train_initial_raw = np.einsum("vc,bct->bvt", TIKHONOV_OPERATOR,
                                  train_eeg, optimize=True).astype(np.float32)
    validation_initial_raw = np.einsum("vc,bct->bvt", TIKHONOV_OPERATOR,
                                       validation_eeg, optimize=True).astype(np.float32)
    test_initial_raw = {name: np.einsum("vc,bct->bvt", TIKHONOV_OPERATOR, eeg,
                                        optimize=True).astype(np.float32)
                        for name, (eeg, _) in test_raw.items()}

    input_scale = max(float(np.percentile(np.abs(train_initial_raw), 99.5)), 1e-12)
    target_scale = max(float(np.percentile(np.abs(train_target_raw), 99.5)), 1e-12)
    normalize_initial = lambda x: np.clip(x / input_scale, -10, 10).astype(np.float32)
    normalize_target = lambda x: np.clip(x / target_scale, -10, 10).astype(np.float32)

    train_initial, validation_initial = normalize_initial(train_initial_raw), normalize_initial(validation_initial_raw)
    train_target, validation_target = normalize_target(train_target_raw), normalize_target(validation_target_raw)
    test_initial = {name: normalize_initial(value) for name, value in test_initial_raw.items()}
    test_target = {name: normalize_target(source) for name, (_, source) in test_raw.items()}

    graph_train = PyGDataLoader(SourceDataset(train_initial, train_target,
                                               train_eeg, True),
                                batch_size=RUN["batch_size"], shuffle=True,
                                num_workers=0)
    graph_validation = PyGDataLoader(SourceDataset(validation_initial,
                                                    validation_target,
                                                    validation_eeg, True),
                                     batch_size=RUN["batch_size"], shuffle=False,
                                     num_workers=0)
    plain_train = PyGDataLoader(SourceDataset(train_initial, train_target,
                                               train_eeg, False),
                                batch_size=RUN["batch_size"], shuffle=True,
                                num_workers=0)
    plain_validation = PyGDataLoader(SourceDataset(validation_initial,
                                                    validation_target,
                                                    validation_eeg, False),
                                     batch_size=RUN["batch_size"], shuffle=False,
                                     num_workers=0)

    models = {"tikhonov_cnn": TikhonovTemporalCNN().to(DEVICE),
              "sparse_graph": SparseGraphNet().to(DEVICE),
              "physics_gat": PhysicsGAT().to(DEVICE)}

    if seed_index == 0:
        smoke_batch = next(iter(graph_train))
        initial, target, eeg = reshape_batch(smoke_batch)
        prediction = models["physics_gat"](smoke_batch.to(DEVICE))
        loss, components = composite_loss(prediction, target, eeg, target_scale)
        assert_finite("smoke prediction", prediction)
        assert_finite("smoke loss", loss)
        for key, value in components.items():
            assert_finite(f"smoke {key}", value)
        loss.backward()
        models["physics_gat"].zero_grad(set_to_none=True)
        print("Smoke test passed")

    trained = {}
    for name, model in models.items():
        train_loader = graph_train if name == "physics_gat" else plain_train
        validation_loader = graph_validation if name == "physics_gat" else plain_validation
        model, history, minutes = train_model(
            name, model, train_loader, validation_loader,
            OUT / f"best_{name}_seed_{seed}.pt", target_scale,
            run_hash(seed),
        )
        trained[name] = model
        history.to_csv(OUT / f"history_{name}_seed_{seed}.csv", index=False)

    convdip_model = None
    if CFG.train_convdip:
        seed_all(seed)
        convdip_model = Net(fwd_fixed)
        # Skip ConvDip in smoke tests due to esinet validation scaling/dimension bugs
        # on very tiny sample sets, but run it in pilot/full mode.
        if CFG.run_mode != "smoke":
            convdip_model.fit(train_sim, epochs=RUN["epochs"],
                              batch_size=RUN["batch_size"])
            try:
                convdip_dir = OUT / f"convdip_seed_{seed}"
                convdip_dir.mkdir(parents=True, exist_ok=True)
                convdip_model.save(str(convdip_dir))
            except Exception as error:
                warnings.warn(f"ConvDip save failed: {error}")
        else:
            convdip_model = None

    for condition, (physical_eeg, target_raw) in test_raw.items():
        initial = test_initial[condition]
        target_normalized = test_target[condition]
        predictions = {}
        source_tensor = torch.from_numpy(initial).to(DEVICE)
        with torch.no_grad():
            predictions["tikhonov_cnn"] = trained["tikhonov_cnn"](source_tensor).cpu().numpy() * target_scale
            predictions["sparse_graph"] = trained["sparse_graph"](source_tensor).cpu().numpy() * target_scale
            loader = PyGDataLoader(SourceDataset(initial, target_normalized,
                                                  physical_eeg, True),
                                   batch_size=RUN["batch_size"], shuffle=False,
                                   num_workers=0)
            graph_outputs = [trained["physics_gat"](batch.to(DEVICE)).cpu().numpy()
                             for batch in loader]
            predictions["physics_gat"] = np.concatenate(graph_outputs) * target_scale
        if convdip_model is not None:
            predictions["convdip"] = convdip_synthetic(convdip_model, physical_eeg)

        for algorithm, prediction in predictions.items():
            if prediction.shape != target_raw.shape:
                raise RuntimeError(f"{algorithm} shape mismatch: {prediction.shape}")
            for sample in range(len(target_raw)):
                row = {"seed": seed, "condition": condition,
                       "algorithm": algorithm, "sample": sample}
                row.update(metrics(target_raw[sample], prediction[sample]))
                synthetic_rows.append(row)

    # Real EEG agreement analysis.
    if CFG.run_real_eeg:
        covariance = mne.compute_covariance(epochs, tmin=CFG.epoch_tmin,
                                            tmax=0.0,
                                            method=["shrunk", "empirical"],
                                            verbose=False)
        inverse = mne.minimum_norm.make_inverse_operator(
            epochs.info, fwd_free, covariance, loose=0.2, depth=0.8,
            rank=None, verbose=False)
        for condition, evoked in evokeds.items():
            reference = mne.minimum_norm.apply_inverse(
                evoked, inverse, lambda2=1.0 / max(CFG.train_snr ** 2, 1e-12),
                method="dSPM", verbose=False)
            reference_map = np.max(np.abs(reference.data), axis=1)
            physical_eeg = np.asarray(evoked.data, dtype=np.float32)
            preprocess_start = time.perf_counter()
            initial_raw = TIKHONOV_OPERATOR @ physical_eeg
            initial = np.clip(initial_raw / input_scale, -10, 10).astype(np.float32)
            preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
            source_tensor = torch.from_numpy(initial[None]).to(DEVICE)
            predictions = {}

            for algorithm in ("tikhonov_cnn", "sparse_graph"):
                model = trained[algorithm]
                with torch.no_grad():
                    _ = model(source_tensor)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    output = model(source_tensor)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                predictions[algorithm] = (
                    output.abs().amax(-1)[0].cpu().numpy() * target_scale,
                    preprocess_ms + (time.perf_counter() - started) * 1000.0,
                )

            edge_index, edge_attr = GRAPH_BUILDER(initial)
            gat_batch = next(iter(PyGDataLoader([
                Data(x=torch.from_numpy(initial), edge_index=edge_index,
                     edge_attr=edge_attr)
            ], batch_size=1))).to(DEVICE)
            with torch.no_grad():
                _ = trained["physics_gat"](gat_batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                started = time.perf_counter()
                output = trained["physics_gat"](gat_batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            predictions["physics_gat"] = (
                output.abs().amax(-1)[0].cpu().numpy() * target_scale,
                preprocess_ms + (time.perf_counter() - started) * 1000.0,
            )

            if convdip_model is not None:
                _ = convdip_model.predict(evoked)
                started = time.perf_counter()
                convdip_prediction = convdip_model.predict(evoked)
                convdip_ms = (time.perf_counter() - started) * 1000.0
                convdip_map = np.max(np.abs(convdip_to_array(convdip_prediction)[0]), axis=1)
                predictions["convdip"] = (convdip_map, convdip_ms)

            for algorithm, (prediction_map, total_ms) in predictions.items():
                reference_norm = normalize_map(reference_map)
                prediction_norm = normalize_map(prediction_map)
                spearman, _ = spearmanr(reference_norm, prediction_norm)
                real_rows.append({
                    "seed": seed, "condition": condition, "algorithm": algorithm,
                    "reference": "dSPM", "interpretation": "agreement_not_ground_truth",
                    "spearman": float(spearman),
                    "cosine": float(1.0 - cosine(reference_norm, prediction_norm)),
                    "map_mse": float(np.mean((reference_norm - prediction_norm) ** 2)),
                    "peak_geodesic_mm": float(GEODESIC_MM[np.argmax(reference_map),
                                                            np.argmax(prediction_map)]),
                })
                timing_rows.append({"seed": seed, "condition": condition,
                                    "algorithm": algorithm, "total_ms": total_ms})

    del train_sim, validation_sim, test_sims, train_eeg, train_target_raw
    del validation_eeg, validation_target_raw, train_initial_raw
    del validation_initial_raw, test_initial_raw, train_initial
    del validation_initial, train_target, validation_target, test_raw
    del test_initial, test_target, trained, models
    if convdip_model is not None:
        del convdip_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# FINAL OUTPUTS AND MANDATORY SANITY CHECKS
# ============================================================================

synthetic_df = pd.DataFrame(synthetic_rows)
real_df = pd.DataFrame(real_rows)
timing_df = pd.DataFrame(timing_rows)

if synthetic_df.empty:
    raise RuntimeError("No synthetic results were generated")
required_algorithms = {"tikhonov_cnn", "sparse_graph", "physics_gat"}
if CFG.train_convdip and CFG.run_mode != "smoke":
    required_algorithms.add("convdip")
missing_algorithms = required_algorithms - set(synthetic_df["algorithm"].unique())
if missing_algorithms:
    raise RuntimeError(f"Missing synthetic algorithms: {sorted(missing_algorithms)}")
required_conditions = {"ID", "OOD_SNR", "OOD_EXTENT", "OOD_SOURCES"}
missing_conditions = required_conditions - set(synthetic_df["condition"].unique())
if missing_conditions:
    raise RuntimeError(f"Missing synthetic conditions: {sorted(missing_conditions)}")
for metric_name in ("waveform_mse", "map_mse", "dle_mm"):
    if not np.isfinite(synthetic_df[metric_name].to_numpy(dtype=float)).all():
        raise RuntimeError(f"Non-finite values found in {metric_name}")
if CFG.run_real_eeg and real_df.empty:
    raise RuntimeError("Real EEG evaluation was enabled but produced no results")

synthetic_df.to_csv(OUT / "synthetic_sample_metrics.csv", index=False)
real_df.to_csv(OUT / "real_eeg_agreement.csv", index=False)
timing_df.to_csv(OUT / "real_eeg_timing.csv", index=False)
synthetic_df.groupby(["seed", "condition", "algorithm"], as_index=False) \
    .mean(numeric_only=True).to_csv(OUT / "synthetic_summary.csv", index=False)

manifest = {
    "configuration": asdict(CFG),
    "resolved_run": RUN,
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda": torch.version.cuda if hasattr(torch.version, 'cuda') else None,
        "torch": torch.__version__,
        "mne": mne.__version__,
        "numpy": np.__version__,
        "esinet": metadata.version("esinet") if "esinet" in [d.metadata["Name"] for d in metadata.distributions()] else "unknown",
        "torch_geometric": metadata.version("torch_geometric") if "torch_geometric" in [d.metadata["Name"] for d in metadata.distributions()] else "unknown",
        "ram_gb": psutil.virtual_memory().total / 1024 ** 3,
    },
    "scientific_warnings": [
        "Real EEG evaluation measures agreement with dSPM, not ground-truth accuracy.",
        "OOD SNR, extent, and source-count tests are distribution shifts.",
        "Physical generalization still requires forward-model and sensor-layout mismatch tests.",
        "ConvDip validation behavior depends on the installed ESINet version.",
    ],
}
with open(OUT / "manifest.json", "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, ensure_ascii=False, indent=2, default=str)

print("FINAL SANITY CHECKS PASSED")
print("Pipeline completed. Results saved to:", OUT)
