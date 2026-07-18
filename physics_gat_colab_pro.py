from __future__ import annotations
# -*- coding: utf-8 -*-
"""
PHYSICS-GAT FINAL PIPELINE — COLAB PRO SINGLE-CELL VERSION
==========================================================
This is the combined, streamlined code for Colab Pro execution.
It includes all 3 parts of the original script, with Google Drive
integration and deep profiling for Colab Pro environments (V100/A100/L4).
"""

# %% 0 — INSTALL IN A SEPARATE COLAB CELL
import os
try:
    import IPython
    IPython.get_ipython().system('pip install -q "mne>=1.7,<1.10" "torch-geometric>=2.5,<2.7" scipy scikit-learn statsmodels pandas matplotlib seaborn tqdm')
except Exception:
    pass

# %% 1 — IMPORTS AND DRIVE MOUNT

import gc
import json
import math
import random
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal, Optional, Tuple
import psutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import butter, sosfiltfilt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from statsmodels.stats.multitest import multipletests

import mne
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATv2Conv, GCNConv, SAGPooling
from torch_geometric.utils import add_self_loops, coalesce, subgraph

mne.set_log_level("WARNING")

# Mount Google Drive for persistent storage
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted successfully.")
except ImportError:
    print("Not running in Google Colab. Skipping Google Drive mount.")

# DEEP ASSESS PROFILING
def print_deep_assessment(stage_name: str):
    print(f"\n--- Deep Assessment: {stage_name} ---")
    ram_info = psutil.virtual_memory()
    print(f"System RAM: {ram_info.used / (1024**3):.2f} GB used / {ram_info.total / (1024**3):.2f} GB total ({ram_info.percent}%)")
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated() / (1024**3)
        vram_reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"GPU VRAM: {vram_allocated:.2f} GB allocated / {vram_reserved:.2f} GB reserved")
        print(f"Max VRAM Allocated: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
    print("-" * 40 + "\n")


# %% 2 — CONFIGURATION FOR COLAB PRO
@dataclass
class Config:
    seed: int = 42
    output_dir: str = "/content/drive/MyDrive/physics_gat_final"
    reset_output_dir: bool = False
    smoke_test: bool = False

    # MNE and Source Generation
    source_spacing: str = "ico3"
    sfreq: float = 100.0
    duration_s: float = 0.30
    mindist_mm: float = 5.0

    # Dataset Sizes
    n_train: int = 8000
    n_validation: int = 1000
    n_test: int = 2000

    train_sources: Tuple[int, int] = (1, 5)
    ood_sources: Tuple[int, int] = (6, 8)
    train_sigma_mm: Tuple[float, float] = (5.0, 25.0)
    ood_sigma_mm: Tuple[float, float] = (28.0, 42.0)
    train_snr_db: Tuple[float, float] = (-5.0, 10.0)
    ood_snr_db: Tuple[float, float] = (-15.0, -7.0)
    amplitude_range: Tuple[float, float] = (0.5, 2.0)
    minimum_center_distance_mm: float = 25.0

    tikhonov_relative: float = 1e-2
    nonlocal_k: int = 4
    functional_k: int = 3

    # Graph
    threshold_ratio: float = 0.04
    minimum_keep_ratio: float = 0.20
    threshold_dilation_hops: int = 1

    hidden: int = 32
    heads: int = 4
    dropout: float = 0.15

    # Hardware Tuning for Colab Pro (A100 / High-RAM)
    batch_size: int = 16
    workers: int = 4

    atlas_mode: Literal["none", "freesurfer", "precomputed"] = "none"
    vertex_roi_file: Optional[str] = None

    # Part 2 Extensions
    sag_ratio: float = 0.50
    cnn_rank: int = 8

    waveform_weight: float = 1.0
    map_weight: float = 1.0
    geodesic_weight: float = 2e-3
    smoothness_weight: float = 2e-4
    sparsity_weight: float = 1e-6
    active_ratio: float = 0.10
    active_weight: float = 12.0

    # Part 3 Training
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    accumulation_steps: int = 2
    scheduler_patience: int = 5
    early_stopping: int = 15
    amp: bool = True
    seeds: Tuple[int, ...] = (42, 52, 62, 72, 82)

    peak_ratio: float = 0.25
    peak_separation_mm: float = 20.0
    maximum_peaks: int = 8
    unmatched_penalty_mm: float = 100.0
    run_real_eeg: bool = False


CFG = Config()

if CFG.smoke_test:
    CFG.n_train = 8
    CFG.n_validation = 4
    CFG.n_test = 4
    CFG.batch_size = 1
    CFG.workers = 0
    CFG.epochs = 1
    CFG.accumulation_steps = 2
    CFG.seeds = (42,)

OUT = Path(CFG.output_dir)
if CFG.reset_output_dir and OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM GB:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    )


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_all(CFG.seed)

with open(OUT / "configuration.json", "w", encoding="utf-8") as handle:
    json.dump(asdict(CFG), handle, ensure_ascii=False, indent=2)

print_deep_assessment("Initialization and Setup")


# %% 3 — MNE SAMPLE FORWARD MODEL
def prepare_mne_sample():
    data_path = Path(mne.datasets.sample.data_path(verbose=False))
    subjects_dir = data_path / "subjects"
    sample_dir = data_path / "MEG" / "sample"

    raw_path = next(
        (
            path
            for path in (
                sample_dir / "sample_audvis_filt-0-40_raw.fif",
                sample_dir / "sample_audvis_raw.fif",
            )
            if path.exists()
        ),
        None,
    )
    if raw_path is None:
        raise FileNotFoundError("MNE sample raw file was not found")

    trans_path = sample_dir / "sample_audvis_raw-trans.fif"
    if not trans_path.exists():
        raise FileNotFoundError(f"Transform file was not found: {trans_path}")

    bem_path = next(
        (
            path
            for path in (
                subjects_dir
                / "sample"
                / "bem"
                / "sample-5120-5120-5120-bem-sol.fif",
                sample_dir / "sample-5120-5120-5120-bem-sol.fif",
            )
            if path.exists()
        ),
        None,
    )
    if bem_path is None:
        raise FileNotFoundError("MNE sample BEM solution was not found")

    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
    raw.filter(1.0, 40.0, picks="eeg", verbose=False)
    raw.set_eeg_reference("average", projection=True, verbose=False)
    raw.apply_proj()

    eeg_raw = raw.copy().pick("eeg")

    source_space = mne.setup_source_space(
        "sample",
        spacing=CFG.source_spacing,
        subjects_dir=str(subjects_dir),
        add_dist=False,
        verbose=False,
    )

    forward_free = mne.make_forward_solution(
        eeg_raw.info,
        trans=str(trans_path),
        src=source_space,
        bem=str(bem_path),
        meg=False,
        eeg=True,
        mindist=CFG.mindist_mm,
        n_jobs=1,
        verbose=False,
    )
    forward_free = mne.pick_types_forward(
        forward_free,
        meg=False,
        eeg=True,
        ref_meg=False,
        exclude="bads",
    )

    forward_fixed = mne.convert_forward_solution(
        forward_free,
        surf_ori=True,
        force_fixed=True,
        use_cps=True,
        copy=True,
        verbose=False,
    )

    lead_field = np.asarray(
        forward_fixed["sol"]["data"], dtype=np.float32
    )
    coordinates_mm = (
        np.vstack(
            [
                source["rr"][source["vertno"]]
                for source in forward_fixed["src"]
            ]
        ).astype(np.float32)
        * 1000.0
    )

    forward_channels = list(
        forward_fixed["sol"]["row_names"]
    )

    if len(forward_channels) != lead_field.shape[0]:
        raise RuntimeError(
            "Forward channel names and lead-field rows are inconsistent: "
            f"names={len(forward_channels)}, "
            f"rows={lead_field.shape[0]}"
        )

    missing_channels = [
        channel
        for channel in forward_channels
        if channel not in eeg_raw.ch_names
    ]

    if missing_channels:
        raise RuntimeError(
            "EEG data are missing channels required by the forward model: "
            f"{missing_channels}"
        )

    eeg_raw = eeg_raw.copy().pick(
        forward_channels
    )
    eeg_raw.reorder_channels(
        forward_channels
    )

    if list(eeg_raw.ch_names) != forward_channels:
        raise RuntimeError(
            "Failed to align EEG channels with the lead-field row order"
        )

    if eeg_raw.info["nchan"] != lead_field.shape[0]:
        raise RuntimeError(
            "Aligned EEG channel count does not match the lead field: "
            f"EEG={eeg_raw.info['nchan']}, "
            f"lead_field={lead_field.shape[0]}"
        )

    print(
        "[OK] EEG and lead-field channels aligned:",
        len(forward_channels),
    )

    return {
        "raw": raw,
        "eeg_info": eeg_raw.info.copy(),
        "subjects_dir": subjects_dir,
        "forward_free": forward_free,
        "forward_fixed": forward_fixed,
        "lead_field": lead_field,
        "coordinates_mm": coordinates_mm,
        "forward_channels": forward_channels,
    }


MNE_OBJECTS = prepare_mne_sample()

RAW = MNE_OBJECTS["raw"]
EEG_INFO = MNE_OBJECTS["eeg_info"]
SUBJECTS_DIR = MNE_OBJECTS["subjects_dir"]
FWD_FREE = MNE_OBJECTS["forward_free"]
FWD_FIXED = MNE_OBJECTS["forward_fixed"]
LEAD_FIELD = MNE_OBJECTS["lead_field"]
COORDINATES_MM = MNE_OBJECTS["coordinates_mm"]
FORWARD_CHANNELS = MNE_OBJECTS["forward_channels"]

N_CHANNELS, N_VERTICES = LEAD_FIELD.shape
N_TIMES = int(round(CFG.duration_s * CFG.sfreq)) + 1

print("Lead field:", LEAD_FIELD.shape)
print("Forward EEG channels:", len(FORWARD_CHANNELS))
print("Source vertices:", N_VERTICES)
print("Time samples:", N_TIMES)


# %% 4 — CORTICAL MESH AND GEODESICS
def extract_cortical_edges(forward_model) -> np.ndarray:
    edge_set = set()
    vertex_offset = 0

    for hemisphere in forward_model["src"]:
        used_vertices = np.asarray(hemisphere["vertno"], dtype=np.int64)
        vertex_lookup = {
            int(vertex): index
            for index, vertex in enumerate(used_vertices)
        }

        triangles = hemisphere.get("use_tris", None)
        if triangles is None:
            triangles = hemisphere.get("tris", None)
        if triangles is None:
            raise RuntimeError("No cortical triangles were found")

        for triangle in np.asarray(triangles, dtype=np.int64):
            if not all(int(vertex) in vertex_lookup for vertex in triangle):
                continue

            nodes = [
                vertex_lookup[int(vertex)] + vertex_offset
                for vertex in triangle
            ]

            for first, second in (
                (nodes[0], nodes[1]),
                (nodes[1], nodes[2]),
                (nodes[2], nodes[0]),
            ):
                if first != second:
                    edge_set.add(tuple(sorted((first, second))))

        vertex_offset += len(used_vertices)

    edges = np.asarray(sorted(edge_set), dtype=np.int64).T
    if edges.size == 0:
        raise RuntimeError("The cortical graph is empty")
    return edges


def compute_geodesic_matrix(
    undirected_edges: np.ndarray,
    coordinates_mm: np.ndarray,
) -> np.ndarray:
    first, second = undirected_edges
    edge_lengths = np.linalg.norm(
        coordinates_mm[first] - coordinates_mm[second], axis=1
    )

    sparse_graph = coo_matrix(
        (
            np.concatenate([edge_lengths, edge_lengths]),
            (
                np.concatenate([first, second]),
                np.concatenate([second, first]),
            ),
        ),
        shape=(len(coordinates_mm), len(coordinates_mm)),
    ).tocsr()

    distances = np.asarray(
        dijkstra(sparse_graph, directed=False), dtype=np.float32
    )
    finite_values = distances[np.isfinite(distances)]
    if finite_values.size == 0:
        raise RuntimeError("No finite geodesic distances were computed")
    distances[~np.isfinite(distances)] = finite_values.max() + 100.0
    return distances


UNDIRECTED_EDGES = extract_cortical_edges(FWD_FIXED)
GEODESIC_MM = compute_geodesic_matrix(UNDIRECTED_EDGES, COORDINATES_MM)
np.save(OUT / "geodesic_mm.npy", GEODESIC_MM)


def make_local_edge_index() -> torch.Tensor:
    first, second = UNDIRECTED_EDGES
    edge_index = torch.tensor(
        np.vstack(
            [
                np.concatenate([first, second]),
                np.concatenate([second, first]),
            ]
        ),
        dtype=torch.long,
    )
    edge_index, _ = add_self_loops(edge_index, num_nodes=N_VERTICES)
    return coalesce(edge_index, num_nodes=N_VERTICES)


LOCAL_EDGE_INDEX = make_local_edge_index()
print_deep_assessment("MNE & Graph Setup")


# %% 5 — LEAD-FIELD PROFILES AND NON-LOCAL EDGES
def normalized_leadfield_profiles(lead_field: np.ndarray) -> np.ndarray:
    profiles = np.asarray(lead_field.T, dtype=np.float32).copy()
    profiles -= profiles.mean(axis=1, keepdims=True)
    profiles /= np.linalg.norm(profiles, axis=1, keepdims=True) + 1e-8
    return profiles


def create_base_graph(lead_field: np.ndarray):
    profiles = normalized_leadfield_profiles(lead_field)

    similarity = np.abs(profiles @ profiles.T)
    np.fill_diagonal(similarity, -np.inf)

    neighbours = np.argpartition(
        similarity,
        -CFG.nonlocal_k,
        axis=1,
    )[:, -CFG.nonlocal_k:]

    source = np.repeat(np.arange(N_VERTICES), CFG.nonlocal_k)
    target = neighbours.reshape(-1)
    nonlocal_edges = torch.tensor(
        np.vstack([source, target]), dtype=torch.long
    )

    edge_index = torch.cat(
        [
            LOCAL_EDGE_INDEX,
            nonlocal_edges,
            nonlocal_edges.flip(0),
        ],
        dim=1,
    )
    edge_index = coalesce(edge_index, num_nodes=N_VERTICES)
    return edge_index, profiles


def create_edge_attributes(
    edge_index: torch.Tensor,
    leadfield_profiles: np.ndarray,
    functional_similarity: Optional[np.ndarray] = None,
) -> torch.Tensor:
    first = edge_index[0].cpu().numpy()
    second = edge_index[1].cpu().numpy()

    euclidean = np.linalg.norm(
        COORDINATES_MM[first] - COORDINATES_MM[second],
        axis=1,
    ) / 200.0

    geodesic = GEODESIC_MM[first, second] / 300.0

    leadfield_similarity = np.abs(
        np.sum(
            leadfield_profiles[first] * leadfield_profiles[second],
            axis=1,
        )
    )

    if functional_similarity is None:
        functional = np.zeros(len(first), dtype=np.float32)
    else:
        functional = functional_similarity[first, second].astype(
            np.float32
        )

    values = np.column_stack(
        [
            euclidean,
            geodesic,
            leadfield_similarity,
            functional,
        ]
    ).astype(np.float32)

    return torch.from_numpy(values)


# %% 6 — STATIC AND DYNAMIC GRAPH BUILDERS
class GraphBuilder:
    def __init__(self, lead_field: np.ndarray, dynamic: bool):
        self.dynamic = bool(dynamic)
        self.base_edges, self.profiles = create_base_graph(lead_field)
        self.static_attributes = create_edge_attributes(
            self.base_edges,
            self.profiles,
        )

    def __call__(
        self,
        initial_source: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if initial_source.shape != (N_VERTICES, N_TIMES):
            raise ValueError(
                "initial_source must have shape "
                f"({N_VERTICES}, {N_TIMES}), got {initial_source.shape}"
            )

        if not self.dynamic:
            return self.base_edges, self.static_attributes

        signal = initial_source - initial_source.mean(
            axis=1, keepdims=True
        )
        signal /= np.linalg.norm(signal, axis=1, keepdims=True) + 1e-8

        functional_similarity = np.abs(signal @ signal.T)
        np.fill_diagonal(functional_similarity, -np.inf)

        neighbours = np.argpartition(
            functional_similarity,
            -CFG.functional_k,
            axis=1,
        )[:, -CFG.functional_k:]

        source = np.repeat(np.arange(N_VERTICES), CFG.functional_k)
        target = neighbours.reshape(-1)
        functional_edges = torch.tensor(
            np.vstack([source, target]), dtype=torch.long
        )

        final_edges = torch.cat(
            [
                self.base_edges,
                functional_edges,
                functional_edges.flip(0),
            ],
            dim=1,
        )
        final_edges = coalesce(final_edges, num_nodes=N_VERTICES)

        final_attributes = create_edge_attributes(
            final_edges,
            self.profiles,
            functional_similarity,
        )
        return final_edges, final_attributes


STATIC_GRAPH_BUILDER = GraphBuilder(LEAD_FIELD, dynamic=False)
DYNAMIC_GRAPH_BUILDER = GraphBuilder(LEAD_FIELD, dynamic=True)


# %% 7 — GAUSSIAN EXTENDED-SOURCE SIMULATOR
def select_separated_centers(rng, number_of_sources):
    selected = []
    for candidate in rng.permutation(N_VERTICES):
        if not selected or np.all(
            GEODESIC_MM[candidate, selected]
            >= CFG.minimum_center_distance_mm
        ):
            selected.append(int(candidate))
        if len(selected) == number_of_sources:
            break

    if len(selected) != number_of_sources:
        raise RuntimeError(
            "Unable to choose the requested number of separated sources"
        )
    return np.asarray(selected, dtype=np.int64)


def generate_waveform(rng):
    time_axis = np.arange(N_TIMES, dtype=np.float32) / CFG.sfreq
    mode = int(rng.integers(0, 4))

    if mode == 0:
        center = rng.uniform(0.07, 0.24)
        width = rng.uniform(0.018, 0.06)
        waveform = np.exp(-0.5 * ((time_axis - center) / width) ** 2)

    elif mode == 1:
        frequency = rng.uniform(4.0, 22.0)
        center = rng.uniform(0.08, 0.23)
        width = rng.uniform(0.035, 0.09)
        waveform = np.exp(-0.5 * ((time_axis - center) / width) ** 2)
        waveform *= np.sin(
            2.0 * np.pi * frequency * time_axis
            + rng.uniform(0.0, 2.0 * np.pi)
        )

    elif mode == 2:
        frequency = rng.uniform(6.0, 28.0)
        onset = rng.uniform(0.02, 0.12)
        decay = rng.uniform(7.0, 18.0)
        shifted = np.maximum(time_axis - onset, 0.0)
        waveform = (time_axis >= onset) * np.exp(-decay * shifted)
        waveform *= np.sin(2.0 * np.pi * frequency * shifted)

    else:
        raw = rng.standard_normal(N_TIMES)
        sos = butter(
            3,
            [2.0, min(30.0, 0.45 * CFG.sfreq)],
            btype="bandpass",
            fs=CFG.sfreq,
            output="sos",
        )
        waveform = sosfiltfilt(sos, raw)

    waveform = np.asarray(waveform, dtype=np.float32)
    waveform /= np.max(np.abs(waveform)) + 1e-8
    return -waveform if rng.random() < 0.5 else waveform


def add_noise_at_snr(rng, clean_eeg, snr_db):
    mode = int(rng.integers(0, 4))

    if mode == 0:
        noise = rng.standard_normal(clean_eeg.shape)

    elif mode == 1:
        frequencies = np.fft.rfftfreq(N_TIMES)
        scale = np.ones_like(frequencies)
        scale[1:] = 1.0 / np.sqrt(frequencies[1:])
        spectrum = (
            rng.standard_normal((clean_eeg.shape[0], len(frequencies)))
            + 1j
            * rng.standard_normal((clean_eeg.shape[0], len(frequencies)))
        )
        noise = np.fft.irfft(
            spectrum * scale[None, :],
            n=N_TIMES,
            axis=1,
        ).real

    elif mode == 2:
        noise = rng.standard_normal(clean_eeg.shape)
        noise *= rng.lognormal(
            0.0, 0.5, (clean_eeg.shape[0], 1)
        )

    else:
        common = rng.standard_normal((1, N_TIMES))
        noise = 0.6 * common + 0.4 * rng.standard_normal(
            clean_eeg.shape
        )

    clean_power = np.mean(clean_eeg**2)
    noise_power = np.mean(noise**2)
    multiplier = np.sqrt(
        clean_power
        / (noise_power * 10.0 ** (snr_db / 10.0) + 1e-30)
    )
    return (clean_eeg + multiplier * noise).astype(np.float32)


def simulate_trial(
    seed,
    lead_field,
    source_range,
    sigma_range,
    snr_range,
):
    rng = np.random.default_rng(seed)
    number_of_sources = int(
        rng.integers(source_range[0], source_range[1] + 1)
    )
    centers = select_separated_centers(rng, number_of_sources)

    source = np.zeros((N_VERTICES, N_TIMES), dtype=np.float32)
    for center in centers:
        sigma = rng.uniform(*sigma_range)
        amplitude = rng.uniform(*CFG.amplitude_range)
        spatial_profile = np.exp(
            -(GEODESIC_MM[center] ** 2) / (2.0 * sigma**2)
        ).astype(np.float32)
        spatial_profile /= spatial_profile.max() + 1e-8
        source += (
            amplitude
            * spatial_profile[:, None]
            * generate_waveform(rng)[None, :]
        )

    clean_eeg = lead_field @ source
    snr_db = float(rng.uniform(*snr_range))
    noisy_eeg = add_noise_at_snr(rng, clean_eeg, snr_db)
    return noisy_eeg, source, centers, snr_db


# %% 8 — TIKHONOV INITIALIZATION AND DATASETS
def make_tikhonov_operator(lead_field):
    matrix = np.asarray(lead_field, dtype=np.float64)
    gram = matrix @ matrix.T
    regularization = CFG.tikhonov_relative * max(
        np.trace(gram) / gram.shape[0], 1e-30
    )
    operator = np.linalg.solve(
        gram + regularization * np.eye(gram.shape[0]),
        matrix,
    ).T
    return operator.astype(np.float32)


TRAIN_OPERATOR = make_tikhonov_operator(LEAD_FIELD)


class SourceDataset(Dataset):
    def __init__(
        self,
        size,
        base_seed,
        lead_field,
        inverse_operator,
        graph_builder,
        source_range,
        sigma_range,
        snr_range,
        scales=None,
    ):
        self.size = int(size)
        self.base_seed = int(base_seed)
        self.lead_field = lead_field
        self.inverse_operator = inverse_operator
        self.graph_builder = graph_builder
        self.source_range = source_range
        self.sigma_range = sigma_range
        self.snr_range = snr_range
        self.scales = scales

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        eeg, source, centers, snr_db = simulate_trial(
            self.base_seed + int(index),
            self.lead_field,
            self.source_range,
            self.sigma_range,
            self.snr_range,
        )
        initial_source = self.inverse_operator @ eeg

        if self.scales is not None:
            eeg_scale, initial_scale, source_scale = self.scales
            eeg = np.clip(eeg / eeg_scale, -10.0, 10.0)
            initial_source = np.clip(
                initial_source / initial_scale, -10.0, 10.0
            )
            source = np.clip(source / source_scale, -10.0, 10.0)

        edge_index, edge_attr = self.graph_builder(initial_source)

        return {
            "eeg": torch.from_numpy(eeg.astype(np.float32)),
            "initial": torch.from_numpy(
                initial_source.astype(np.float32)
            ),
            "source": torch.from_numpy(source.astype(np.float32)),
            "centers": torch.from_numpy(centers),
            "snr": torch.tensor(snr_db, dtype=torch.float32),
            "edge_index": edge_index,
            "edge_attr": edge_attr,
        }


def estimate_training_scales(dataset, samples=128):
    values = {"eeg": [], "initial": [], "source": []}
    for index in range(min(samples, len(dataset))):
        item = dataset[index]
        for key in values:
            values[key].append(
                np.abs(item[key].numpy()).reshape(-1)
            )

    return tuple(
        max(
            float(
                np.percentile(
                    np.concatenate(values[key]),
                    99.5,
                )
            ),
            1e-12,
        )
        for key in ("eeg", "initial", "source")
    )


RAW_TRAIN_DATASET = SourceDataset(
    size=CFG.n_train,
    base_seed=100_000,
    lead_field=LEAD_FIELD,
    inverse_operator=TRAIN_OPERATOR,
    graph_builder=STATIC_GRAPH_BUILDER,
    source_range=CFG.train_sources,
    sigma_range=CFG.train_sigma_mm,
    snr_range=CFG.train_snr_db,
    scales=None,
)

SCALES = estimate_training_scales(RAW_TRAIN_DATASET)
print("Training scales:", SCALES)


def make_dataset(
    split,
    condition="id",
    dynamic_graph=False,
    lead_field=LEAD_FIELD,
    inverse_operator=TRAIN_OPERATOR,
):
    if split == "train":
        size, base_seed = CFG.n_train, 100_000
    elif split == "validation":
        size, base_seed = CFG.n_validation, 200_000
    elif split == "test":
        size, base_seed = CFG.n_test, 300_000
    else:
        raise ValueError(f"Unknown split: {split}")

    source_range = CFG.train_sources
    sigma_range = CFG.train_sigma_mm
    snr_range = CFG.train_snr_db

    if condition == "ood_snr":
        snr_range = CFG.ood_snr_db
        base_seed += 10_000
    elif condition == "ood_sources":
        source_range = CFG.ood_sources
        base_seed += 20_000
    elif condition == "ood_extent":
        sigma_range = CFG.ood_sigma_mm
        base_seed += 30_000
    elif condition != "id":
        raise ValueError(f"Unknown condition: {condition}")

    graph_builder = GraphBuilder(
        lead_field,
        dynamic=dynamic_graph,
    )

    return SourceDataset(
        size=size,
        base_seed=base_seed,
        lead_field=lead_field,
        inverse_operator=inverse_operator,
        graph_builder=graph_builder,
        source_range=source_range,
        sigma_range=sigma_range,
        snr_range=snr_range,
        scales=SCALES,
    )


def collate_trials(items):
    return {
        "eeg": torch.stack([item["eeg"] for item in items]),
        "initial": torch.stack(
            [item["initial"] for item in items]
        ),
        "source": torch.stack([item["source"] for item in items]),
        "centers": [item["centers"] for item in items],
        "snr": torch.stack([item["snr"] for item in items]),
        "edge_index": [item["edge_index"] for item in items],
        "edge_attr": [item["edge_attr"] for item in items],
    }


def make_loader(dataset, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=shuffle,
        num_workers=CFG.workers,
        persistent_workers=CFG.workers > 0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_trials,
    )

print_deep_assessment("Dataset Creation")


# %% 9 — ENCODERS & GAT BLOCKS
class TemporalEncoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.GroupNorm(2, 8),
            nn.GELU(),
            nn.Conv1d(8, 12, kernel_size=3, padding=1),
            nn.GroupNorm(3, 12),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.projection = nn.Linear(48, hidden)

    def forward(self, signal):
        if signal.ndim != 2 or signal.shape[1] != N_TIMES:
            raise ValueError(
                f"Expected [nodes, {N_TIMES}], got {signal.shape}"
            )
        features = self.network(signal[:, None, :]).flatten(1)
        return self.projection(features)


class EdgeAwareGATBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        if hidden % CFG.heads != 0:
            raise ValueError("hidden must be divisible by heads")

        self.gat = GATv2Conv(
            hidden,
            hidden // CFG.heads,
            heads=CFG.heads,
            edge_dim=4,
            dropout=CFG.dropout,
            residual=True,
            add_self_loops=False,
        )
        self.normalization = nn.LayerNorm(hidden)

    def forward(self, features, edge_index, edge_attr):
        if edge_attr.shape != (edge_index.shape[1], 4):
            raise ValueError(
                "edge_attr must have one four-dimensional row per edge"
            )
        message = self.gat(features, edge_index, edge_attr)
        message = F.dropout(
            F.gelu(message),
            p=CFG.dropout,
            training=self.training,
        )
        return self.normalization(features + message)


class PhysicsGATEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(CFG.hidden)
        self.graph_block_1 = EdgeAwareGATBlock(CFG.hidden)
        self.graph_block_2 = EdgeAwareGATBlock(CFG.hidden)

    def forward(self, initial_source, edge_index, edge_attr):
        features = self.temporal_encoder(initial_source)
        features = self.graph_block_1(
            features, edge_index, edge_attr
        )
        features = self.graph_block_2(
            features, edge_index, edge_attr
        )
        return features


# %% 10 — CONSERVATIVE THRESHOLD AND TRUE INDUCED SUBGRAPH
def make_threshold_subgraph(
    source_signal: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
):
    if source_signal.ndim != 2:
        raise ValueError("source_signal must have shape [nodes, times]")

    number_of_nodes = source_signal.shape[0]
    activity = source_signal.abs().amax(dim=-1)
    threshold = (
        CFG.threshold_ratio * activity.max().clamp_min(1e-8)
    )
    keep_mask = activity >= threshold

    minimum_nodes = max(
        1,
        math.ceil(number_of_nodes * CFG.minimum_keep_ratio),
    )
    top_indices = activity.topk(minimum_nodes).indices
    keep_mask[top_indices] = True

    for _ in range(CFG.threshold_dilation_hops):
        first, second = edge_index
        expanded = keep_mask.clone()
        expanded[second] |= keep_mask[first]
        keep_mask = expanded

    selected_nodes = torch.where(keep_mask)[0]
    if len(selected_nodes) == 0:
        selected_nodes = activity.argmax().reshape(1)

    sub_edge_index, sub_edge_attr, edge_mask = subgraph(
        subset=selected_nodes,
        edge_index=edge_index,
        edge_attr=edge_attr,
        relabel_nodes=True,
        num_nodes=number_of_nodes,
        return_edge_mask=True,
    )

    if sub_edge_attr is None:
        raise RuntimeError("Subgraph edge attributes were lost")
    if sub_edge_attr.shape[0] != sub_edge_index.shape[1]:
        raise RuntimeError("Subgraph edges and attributes are inconsistent")

    return {
        "selected_nodes": selected_nodes,
        "edge_index": sub_edge_index,
        "edge_attr": sub_edge_attr,
        "edge_mask": edge_mask,
        "keep_mask": keep_mask,
    }


# %% 13 — FULL VERTEX-LEVEL PHYSICS-GAT
class FullPhysicsGAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(CFG.hidden)
        self.graph_block_1 = EdgeAwareGATBlock(CFG.hidden)
        self.graph_block_2 = EdgeAwareGATBlock(CFG.hidden)
        self.temporal_decoder = nn.Sequential(
            nn.Linear(CFG.hidden, CFG.hidden),
            nn.GELU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(CFG.hidden, N_TIMES),
        )

    def forward_one(self, initial_source, edge_index, edge_attr):
        features = self.temporal_encoder(initial_source)
        features = self.graph_block_1(features, edge_index, edge_attr)
        features = self.graph_block_2(features, edge_index, edge_attr)
        return self.temporal_decoder(features)

    def forward(self, initial_source, edge_indices, edge_attributes):
        if initial_source.ndim != 3:
            raise ValueError("initial_source must have shape [B,V,T]")
        outputs = []
        for sample_index in range(len(initial_source)):
            edge_index = edge_indices[sample_index].to(initial_source.device)
            edge_attr = edge_attributes[sample_index].to(initial_source.device)
            outputs.append(
                self.forward_one(
                    initial_source[sample_index],
                    edge_index,
                    edge_attr,
                )
            )
        return torch.stack(outputs)


# %% 14 — THRESHOLD PHYSICS-GAT WITH TRUE SUBGRAPH
class ThresholdPhysicsGAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(CFG.hidden)
        self.graph_block_1 = EdgeAwareGATBlock(CFG.hidden)
        self.graph_block_2 = EdgeAwareGATBlock(CFG.hidden)
        self.temporal_decoder = nn.Sequential(
            nn.Linear(CFG.hidden, CFG.hidden),
            nn.GELU(),
            nn.Linear(CFG.hidden, N_TIMES),
        )

    def forward_one(self, initial_source, edge_index, edge_attr):
        thresholded = make_threshold_subgraph(
            initial_source,
            edge_index,
            edge_attr,
        )
        selected_nodes = thresholded["selected_nodes"]
        sub_edges = thresholded["edge_index"]
        sub_attr = thresholded["edge_attr"]

        selected_signal = initial_source[selected_nodes]
        features = self.temporal_encoder(selected_signal)
        features = self.graph_block_1(features, sub_edges, sub_attr)
        features = self.graph_block_2(features, sub_edges, sub_attr)
        selected_output = self.temporal_decoder(features)

        full_output = torch.zeros(
            (N_VERTICES, N_TIMES),
            dtype=selected_output.dtype,
            device=selected_output.device,
        )
        full_output[selected_nodes] = selected_output
        return full_output

    def forward(self, initial_source, edge_indices, edge_attributes):
        outputs = []
        for sample_index in range(len(initial_source)):
            outputs.append(
                self.forward_one(
                    initial_source[sample_index],
                    edge_indices[sample_index].to(initial_source.device),
                    edge_attributes[sample_index].to(initial_source.device),
                )
            )
        return torch.stack(outputs)


# %% 15 — SAGPOOL PHYSICS-GAT
class SAGPoolPhysicsGAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(CFG.hidden)
        self.pre_pool_block = EdgeAwareGATBlock(CFG.hidden)
        self.pool = SAGPooling(
            CFG.hidden,
            ratio=CFG.sag_ratio,
            GNN=GCNConv,
        )
        self.bottleneck_block = EdgeAwareGATBlock(CFG.hidden)
        self.decoder_block_1 = EdgeAwareGATBlock(CFG.hidden)
        self.decoder_block_2 = EdgeAwareGATBlock(CFG.hidden)
        self.temporal_decoder = nn.Sequential(
            nn.Linear(CFG.hidden, CFG.hidden),
            nn.GELU(),
            nn.Linear(CFG.hidden, N_TIMES),
        )
        self.last_pool_statistics = None

    def forward_one(self, initial_source, edge_index, edge_attr):
        number_of_nodes = initial_source.shape[0]
        encoded = self.temporal_encoder(initial_source)
        encoded = self.pre_pool_block(encoded, edge_index, edge_attr)

        batch = torch.zeros(
            number_of_nodes,
            dtype=torch.long,
            device=encoded.device,
        )

        (
            pooled_features,
            pooled_edges,
            pooled_attr,
            pooled_batch,
            permutation,
            scores,
        ) = self.pool(
            encoded,
            edge_index,
            edge_attr=edge_attr,
            batch=batch,
        )

        pooled_features = self.bottleneck_block(
            pooled_features,
            pooled_edges,
            pooled_attr,
        )

        unpooled = torch.zeros_like(encoded)
        unpooled[permutation] = pooled_features

        decoded = self.decoder_block_1(
            encoded + unpooled,
            edge_index,
            edge_attr,
        )
        decoded = self.decoder_block_2(
            decoded,
            edge_index,
            edge_attr,
        )

        self.last_pool_statistics = {
            "nodes_before": int(number_of_nodes),
            "nodes_after": int(len(permutation)),
            "edges_before": int(edge_index.shape[1]),
            "edges_after": int(pooled_edges.shape[1]),
            "mean_score": float(scores.detach().mean().cpu()),
        }
        return self.temporal_decoder(decoded)

    def forward(self, initial_source, edge_indices, edge_attributes):
        outputs = []
        for sample_index in range(len(initial_source)):
            outputs.append(
                self.forward_one(
                    initial_source[sample_index],
                    edge_indices[sample_index].to(initial_source.device),
                    edge_attributes[sample_index].to(initial_source.device),
                )
            )
        return torch.stack(outputs)


# %% 16 — FREESURFER/PRECOMPUTED ROI MAPPING
def make_freesurfer_roi_mapping():
    labels = mne.read_labels_from_annot(
        "sample",
        parc="aparc",
        subjects_dir=str(SUBJECTS_DIR),
        verbose=False,
    )

    roi_index = np.full(N_VERTICES, -1, dtype=np.int64)
    left_count = len(FWD_FIXED["src"][0]["vertno"])
    hemisphere_offsets = [0, left_count]

    for label_index, label in enumerate(labels):
        hemisphere = 0 if label.hemi == "lh" else 1
        used_vertices = np.asarray(
            FWD_FIXED["src"][hemisphere]["vertno"],
            dtype=np.int64,
        )
        lookup = {
            int(vertex): index
            for index, vertex in enumerate(used_vertices)
        }
        offset = hemisphere_offsets[hemisphere]

        for vertex in label.vertices:
            local_index = lookup.get(int(vertex))
            if local_index is not None:
                roi_index[offset + local_index] = label_index

    missing = roi_index < 0
    if np.any(missing):
        fallback_roi = int(roi_index.max()) + 1
        roi_index[missing] = fallback_roi

    unique_values = np.unique(roi_index)
    relabel = {
        int(value): index
        for index, value in enumerate(unique_values)
    }
    return np.asarray(
        [relabel[int(value)] for value in roi_index],
        dtype=np.int64,
    )


def load_roi_mapping():
    if CFG.atlas_mode == "none":
        return None

    if CFG.atlas_mode == "freesurfer":
        return make_freesurfer_roi_mapping()

    if CFG.atlas_mode != "precomputed":
        raise ValueError(f"Unknown atlas mode: {CFG.atlas_mode}")

    if CFG.vertex_roi_file is None:
        raise ValueError(
            "precomputed atlas mode requires CFG.vertex_roi_file"
        )

    mapping = np.load(CFG.vertex_roi_file)
    if mapping.shape != (N_VERTICES,):
        raise ValueError(
            "vertex_roi_file must contain exactly one ROI index per vertex"
        )
    if not np.issubdtype(mapping.dtype, np.integer):
        raise ValueError("ROI mapping must contain integer indices")
    if np.any(mapping < 0):
        raise ValueError("ROI mapping must not contain negative indices")

    unique_values = np.unique(mapping)
    relabel = {
        int(value): index
        for index, value in enumerate(unique_values)
    }
    return np.asarray(
        [relabel[int(value)] for value in mapping],
        dtype=np.int64,
    )


def reduce_vertices_to_rois(vertex_signal, roi_index):
    index = torch.as_tensor(
        roi_index,
        dtype=torch.long,
        device=vertex_signal.device,
    )
    number_of_rois = int(index.max().item()) + 1

    roi_signal = torch.zeros(
        (number_of_rois, vertex_signal.shape[-1]),
        dtype=vertex_signal.dtype,
        device=vertex_signal.device,
    )
    counts = torch.zeros(
        number_of_rois,
        dtype=vertex_signal.dtype,
        device=vertex_signal.device,
    )

    roi_signal.index_add_(0, index, vertex_signal)
    counts.index_add_(
        0,
        index,
        torch.ones(
            len(index),
            dtype=vertex_signal.dtype,
            device=vertex_signal.device,
        ),
    )
    return roi_signal / counts[:, None].clamp_min(1.0)


def expand_rois_to_vertices(roi_signal, roi_index):
    index = torch.as_tensor(
        roi_index,
        dtype=torch.long,
        device=roi_signal.device,
    )
    return roi_signal[index]


def create_roi_edges(roi_index):
    pairs = set()
    first, second = UNDIRECTED_EDGES

    for source_roi, target_roi in zip(
        roi_index[first], roi_index[second]
    ):
        if source_roi != target_roi:
            pairs.add((int(source_roi), int(target_roi)))
            pairs.add((int(target_roi), int(source_roi)))

    if not pairs:
        raise RuntimeError("ROI graph is empty")

    edge_index = torch.tensor(
        np.asarray(sorted(pairs), dtype=np.int64).T,
        dtype=torch.long,
    )
    number_of_rois = int(np.max(roi_index)) + 1
    edge_index, _ = add_self_loops(
        edge_index,
        num_nodes=number_of_rois,
    )
    return coalesce(edge_index, num_nodes=number_of_rois)


ROI_INDEX = load_roi_mapping()


class AtlasPhysicsGAT(nn.Module):
    """ROI graph model with deterministic expansion back to vertex space."""

    def __init__(self, roi_index):
        super().__init__()
        if roi_index is None:
            raise ValueError("AtlasPhysicsGAT requires a valid ROI mapping")

        self.roi_index = np.asarray(roi_index, dtype=np.int64)
        self.roi_edges = create_roi_edges(self.roi_index)
        self.temporal_encoder = TemporalEncoder(CFG.hidden)
        self.graph_block_1 = EdgeAwareGATBlock(CFG.hidden)
        self.graph_block_2 = EdgeAwareGATBlock(CFG.hidden)
        self.temporal_decoder = nn.Linear(CFG.hidden, N_TIMES)

    def forward(self, initial_source, edge_indices, edge_attributes):
        roi_edges = self.roi_edges.to(initial_source.device)
        roi_attributes = torch.zeros(
            (roi_edges.shape[1], 4),
            dtype=initial_source.dtype,
            device=initial_source.device,
        )

        outputs = []
        for sample in initial_source:
            roi_signal = reduce_vertices_to_rois(
                sample, self.roi_index
            )
            features = self.temporal_encoder(roi_signal)
            features = self.graph_block_1(
                features, roi_edges, roi_attributes
            )
            features = self.graph_block_2(
                features, roi_edges, roi_attributes
            )
            roi_output = self.temporal_decoder(features)
            outputs.append(
                expand_rois_to_vertices(
                    roi_output, self.roi_index
                )
            )
        return torch.stack(outputs)


# %% 17 — SIMULATOR METADATA VALIDATION
def inspect_simulation_distribution(number_of_trials=32):
    source_counts = []
    snr_values = []
    active_extent_vertices = []

    for index in range(number_of_trials):
        eeg, source, centers, snr_db = simulate_trial(
            seed=700_000 + index,
            lead_field=LEAD_FIELD,
            source_range=CFG.train_sources,
            sigma_range=CFG.train_sigma_mm,
            snr_range=CFG.train_snr_db,
        )
        source_map_value = np.max(np.abs(source), axis=1)
        threshold = 0.10 * (source_map_value.max() + 1e-12)

        source_counts.append(len(centers))
        snr_values.append(snr_db)
        active_extent_vertices.append(
            int(np.sum(source_map_value >= threshold))
        )

    metadata = {
        "trials": number_of_trials,
        "minimum_sources": int(np.min(source_counts)),
        "maximum_sources": int(np.max(source_counts)),
        "minimum_snr_db": float(np.min(snr_values)),
        "maximum_snr_db": float(np.max(snr_values)),
        "mean_active_extent_vertices": float(
            np.mean(active_extent_vertices)
        ),
    }

    with open(
        OUT / "simulation_distribution.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return metadata

# %% 18 — LOSS TENSORS
SPATIAL_EDGE_TENSOR = torch.tensor(
    np.vstack(
        [
            np.concatenate(
                [UNDIRECTED_EDGES[0], UNDIRECTED_EDGES[1]]
            ),
            np.concatenate(
                [UNDIRECTED_EDGES[1], UNDIRECTED_EDGES[0]]
            ),
        ]
    ),
    dtype=torch.long,
    device=DEVICE,
)

GEODESIC_TENSOR = torch.tensor(
    GEODESIC_MM,
    dtype=torch.float32,
    device=DEVICE,
)

def full_composite_loss(
    prediction,
    target,
    include_geodesic=True,
    include_smoothness=True,
    include_sparsity=True,
):
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: "
            f"{prediction.shape} versus {target.shape}"
        )
    if prediction.ndim != 3:
        raise ValueError("Expected prediction with shape [B,V,T]")

    positive_error = (prediction - target).square().mean(
        dim=(1, 2)
    )
    negative_error = (prediction + target).square().mean(
        dim=(1, 2)
    )
    waveform_loss = torch.minimum(
        positive_error, negative_error
    ).mean()

    predicted_map = prediction.abs().amax(dim=-1)
    target_map = target.abs().amax(dim=-1)

    target_peak = target_map.amax(
        dim=1, keepdim=True
    ).clamp_min(1e-8)
    active_target = target_map >= CFG.active_ratio * target_peak

    map_weights = 1.0 + (
        CFG.active_weight - 1.0
    ) * active_target.float()
    map_loss = (
        map_weights * (predicted_map - target_map).square()
    ).sum() / map_weights.sum().clamp_min(1.0)

    residual_map = predicted_map - target_map
    smoothness_loss = (
        residual_map[:, SPATIAL_EDGE_TENSOR[0]]
        - residual_map[:, SPATIAL_EDGE_TENSOR[1]]
    ).square().mean()

    probability = predicted_map / predicted_map.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-8)

    geodesic_terms = []
    for sample_index in range(len(prediction)):
        true_support = torch.where(active_target[sample_index])[0]
        if len(true_support) == 0:
            true_support = target_map[
                sample_index
            ].argmax().reshape(1)

        distance_to_true_support = GEODESIC_TENSOR[
            :, true_support
        ].amin(dim=1)
        geodesic_terms.append(
            (
                probability[sample_index]
                * distance_to_true_support
            ).sum()
            / 100.0
        )

    geodesic_loss = torch.stack(geodesic_terms).mean()
    sparsity_loss = predicted_map.mean()

    total_loss = (
        CFG.waveform_weight * waveform_loss
        + CFG.map_weight * map_loss
    )
    if include_geodesic:
        total_loss = total_loss + CFG.geodesic_weight * geodesic_loss
    if include_smoothness:
        total_loss = total_loss + CFG.smoothness_weight * smoothness_loss
    if include_sparsity:
        total_loss = total_loss + CFG.sparsity_weight * sparsity_loss

    components = {
        "total": total_loss.detach(),
        "waveform": waveform_loss.detach(),
        "map": map_loss.detach(),
        "geodesic": geodesic_loss.detach(),
        "smoothness": smoothness_loss.detach(),
        "sparsity": sparsity_loss.detach(),
    }
    return total_loss, components


# %% 19 — FAIR LOW-RANK CNN BASELINE
class FairConvDipCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(N_CHANNELS, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Conv1d(96, 96, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
        )
        self.latent = nn.Sequential(
            nn.Linear(96 * 8, 512),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        self.spatial_head = nn.Linear(
            512,
            N_VERTICES * CFG.cnn_rank,
        )
        self.temporal_head = nn.Linear(
            512,
            CFG.cnn_rank * N_TIMES,
        )

    def forward(self, eeg, edge_indices=None, edge_attributes=None):
        if eeg.ndim != 3 or eeg.shape[1:] != (
            N_CHANNELS,
            N_TIMES,
        ):
            raise ValueError(
                "EEG input must have shape "
                f"[B,{N_CHANNELS},{N_TIMES}]"
            )

        latent = self.latent(self.encoder(eeg))
        spatial = self.spatial_head(latent).reshape(
            len(eeg),
            N_VERTICES,
            CFG.cnn_rank,
        )
        temporal = self.temporal_head(latent).reshape(
            len(eeg),
            CFG.cnn_rank,
            N_TIMES,
        )
        return torch.einsum(
            "bvr,brt->bvt",
            spatial,
            temporal,
        )

# %% 22 — METRICS
def extract_separated_peaks(activity):
    activity = np.asarray(activity, dtype=float)
    if activity.shape != (N_VERTICES,):
        raise ValueError(
            f"activity must have shape ({N_VERTICES},), got {activity.shape}"
        )
    if not np.isfinite(activity).all():
        raise ValueError("activity contains non-finite values")

    candidates = np.where(
        activity >= CFG.peak_ratio * (activity.max() + 1e-12)
    )[0]
    candidates = candidates[np.argsort(activity[candidates])[::-1]]

    selected = []
    for candidate in candidates:
        if not selected or np.all(
            GEODESIC_MM[candidate, selected] >= CFG.peak_separation_mm
        ):
            selected.append(int(candidate))
        if len(selected) >= CFG.maximum_peaks:
            break

    if not selected:
        selected = [int(np.argmax(activity))]
    return np.asarray(selected, dtype=np.int64)


def multisource_geodesic_dle(true_centers, predicted_map):
    true_centers = np.asarray(true_centers, dtype=np.int64)
    predicted_centers = extract_separated_peaks(predicted_map)

    if len(true_centers) == 0:
        raise ValueError("At least one true center is required")

    assignment_cost = GEODESIC_MM[
        np.ix_(true_centers, predicted_centers)
    ]
    true_assignment, predicted_assignment = linear_sum_assignment(
        assignment_cost
    )

    distances = assignment_cost[
        true_assignment, predicted_assignment
    ].tolist()
    missed_sources = len(true_centers) - len(true_assignment)
    false_sources = len(predicted_centers) - len(predicted_assignment)
    distances.extend(
        [CFG.unmatched_penalty_mm]
        * (missed_sources + false_sources)
    )

    return {
        "dle_mm": float(np.mean(distances)),
        "missed_sources": int(missed_sources),
        "false_sources": int(false_sources),
        "predicted_sources": int(len(predicted_centers)),
    }


def compute_sample_metrics(target, prediction, true_centers):
    target = np.asarray(target, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    if target.shape != (N_VERTICES, N_TIMES):
        raise ValueError("Target has an invalid shape")
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ")

    target_map = np.max(np.abs(target), axis=1)
    predicted_map = np.max(np.abs(prediction), axis=1)
    target_normalized = target_map / (target_map.max() + 1e-12)
    prediction_normalized = predicted_map / (
        predicted_map.max() + 1e-12
    )

    target_label = (
        target_normalized >= CFG.active_ratio
    ).astype(np.int64)
    predicted_label = (
        prediction_normalized >= CFG.active_ratio
    ).astype(np.int64)

    auc = np.nan
    if np.unique(target_label).size == 2:
        auc = roc_auc_score(target_label, predicted_map)

    precision, recall, f1, _ = precision_recall_fscore_support(
        target_label,
        predicted_label,
        average="binary",
        zero_division=0,
    )

    positive_waveform_error = np.mean((target - prediction) ** 2)
    negative_waveform_error = np.mean((target + prediction) ** 2)
    dle_values = multisource_geodesic_dle(
        true_centers, predicted_map
    )

    result = {
        "waveform_mse_sign_invariant": float(
            min(positive_waveform_error, negative_waveform_error)
        ),
        "map_mse": float(
            np.mean(
                (target_normalized - prediction_normalized) ** 2
            )
        ),
        "cosine": float(
            np.sum(target_normalized * prediction_normalized)
            / (
                np.linalg.norm(target_normalized)
                * np.linalg.norm(prediction_normalized)
                + 1e-12
            )
        ),
        "auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "extent_error_vertices": float(
            abs(predicted_label.sum() - target_label.sum())
        ),
    }
    result.update(dle_values)
    return result

# %% 23 — SENSOR MASKING AND TRUE CHANNEL REMOVAL
class SensorMaskDataset(Dataset):
    """Zeros a fixed sensor subset; graph preprocessing is rebuilt in evaluation."""

    def __init__(self, base_dataset, fraction, seed=9000):
        if not (0.0 < fraction < 1.0):
            raise ValueError("fraction must be in (0,1)")
        self.base_dataset = base_dataset
        self.inverse_operator = base_dataset.inverse_operator
        self.graph_builder = base_dataset.graph_builder

        rng = np.random.default_rng(seed)
        self.dropped_channels = rng.choice(
            N_CHANNELS,
            max(1, round(N_CHANNELS * fraction)),
            replace=False,
        )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        item = self.base_dataset[index]
        eeg = item["eeg"].clone()
        eeg[self.dropped_channels] = 0.0
        item["eeg"] = eeg
        return item


def make_true_channel_removal_dataset(
    fraction=0.25,
    base_seed=600_000,
):
    if not (0.0 < fraction < 1.0):
        raise ValueError("fraction must be in (0,1)")

    rng = np.random.default_rng(123)
    removed_channels = rng.choice(
        N_CHANNELS,
        max(1, round(N_CHANNELS * fraction)),
        replace=False,
    )
    remaining_channels = np.setdiff1d(
        np.arange(N_CHANNELS), removed_channels
    )

    reduced_lead_field = LEAD_FIELD[remaining_channels]
    reduced_operator = make_tikhonov_operator(reduced_lead_field)
    reduced_graph_builder = GraphBuilder(
        reduced_lead_field,
        dynamic=True,
    )

    reduced_dataset = SourceDataset(
        size=CFG.n_test,
        base_seed=base_seed,
        lead_field=reduced_lead_field,
        inverse_operator=reduced_operator,
        graph_builder=reduced_graph_builder,
        source_range=CFG.train_sources,
        sigma_range=CFG.train_sigma_mm,
        snr_range=CFG.train_snr_db,
        scales=SCALES,
    )
    return reduced_dataset, remaining_channels


# %% 24 — LEAD-FIELD OOD
def make_algebraic_leadfield_perturbation(
    scale=0.03,
    seed=777,
):
    rng = np.random.default_rng(seed)
    sensor_transform = np.eye(N_CHANNELS) + (
        scale
        * rng.standard_normal((N_CHANNELS, N_CHANNELS))
        / np.sqrt(N_CHANNELS)
    )
    return (sensor_transform @ LEAD_FIELD).astype(np.float32)


PERTURBED_LEAD_FIELD = make_algebraic_leadfield_perturbation()
PERTURBED_OPERATOR = make_tikhonov_operator(PERTURBED_LEAD_FIELD)


def make_external_forward_ood_dataset(
    lead_field,
    split="test",
    condition="id",
    dynamic_graph=True,
):
    lead_field = np.asarray(lead_field, dtype=np.float32)
    if lead_field.ndim != 2 or lead_field.shape[1] != N_VERTICES:
        raise ValueError(
            f"External lead field must have shape [C,{N_VERTICES}]"
        )
    operator = make_tikhonov_operator(lead_field)
    return make_dataset(
        split=split,
        condition=condition,
        dynamic_graph=dynamic_graph,
        lead_field=lead_field,
        inverse_operator=operator,
    )

# %% 25 — CLASSICAL TIKHONOV BASELINE
def evaluate_tikhonov(dataset):
    rows = []
    processing_times = []

    for index in range(len(dataset)):
        item = dataset[index]
        physical_eeg = item["eeg"].numpy() * SCALES[0]

        start = time.perf_counter()
        prediction = dataset.inverse_operator @ physical_eeg
        processing_times.append(
            (time.perf_counter() - start) * 1000.0
        )
        prediction = prediction / SCALES[2]

        rows.append(
            compute_sample_metrics(
                target=item["source"].numpy(),
                prediction=prediction,
                true_centers=item["centers"].numpy(),
            )
        )

    timing = {
        "physics_ms": float(np.mean(processing_times)),
        "graph_ms": 0.0,
        "network_ms": 0.0,
        "total_ms": float(np.mean(processing_times)),
        "peak_gpu_mb": 0.0,
    }
    return pd.DataFrame(rows), timing


# %% 26 — OFFICIAL CONVDIP ADAPTER INTERFACE
class OfficialConvDipAdapter:
    """Requires an already inspected and verified official ESINet model."""

    def __init__(self, verified_esinet_model=None):
        if verified_esinet_model is None:
            raise RuntimeError(
                "A verified official ESINet/ConvDip model is required. "
                "FairConvDipCNN remains the executable comparison baseline."
            )
        self.model = verified_esinet_model

    def fit(self, simulation, **kwargs):
        return self.model.fit(simulation, **kwargs)

    def predict(self, data):
        return self.model.predict(data)


# %% 27 — CHECKPOINTING, MULTI-SEED, AND TRAINING
def parameter_count(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def save_training_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_validation,
    stale_epochs,
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": int(epoch),
            "best_validation": float(best_validation),
            "stale_epochs": int(stale_epochs),
            "configuration": asdict(CFG),
            "scales": SCALES,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        },
        path,
    )


def train_model(
    model,
    input_kind,
    train_loader,
    validation_loader,
    checkpoint_path,
    include_geodesic=True,
    include_smoothness=True,
    include_sparsity=True,
):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=CFG.scheduler_patience,
        min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=CFG.amp and DEVICE.type == "cuda",
    )

    start_epoch = 1
    best_validation = float("inf")
    stale_epochs = 0

    if checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_validation = checkpoint["best_validation"]
        stale_epochs = checkpoint["stale_epochs"]
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if (
            torch.cuda.is_available()
            and checkpoint["cuda_rng"] is not None
        ):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])

    history = []
    for epoch in range(start_epoch, CFG.epochs + 1):
        epoch_record = {"epoch": epoch}

        for phase, data_loader in (
            ("train", train_loader),
            ("validation", validation_loader),
        ):
            training = phase == "train"
            model.train(training)
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            sample_count = 0

            gradient_context = (
                torch.enable_grad() if training else torch.no_grad()
            )
            with gradient_context:
                for step, batch in enumerate(data_loader):
                    eeg = batch["eeg"].to(
                        DEVICE, non_blocking=True
                    )
                    initial = batch["initial"].to(
                        DEVICE, non_blocking=True
                    )
                    target = batch["source"].to(
                        DEVICE, non_blocking=True
                    )
                    model_input = (
                        eeg if input_kind == "eeg" else initial
                    )

                    with torch.amp.autocast(
                        "cuda",
                        enabled=CFG.amp and DEVICE.type == "cuda",
                    ):
                        prediction = model(
                            model_input,
                            batch["edge_index"],
                            batch["edge_attr"],
                        )
                        loss, _ = full_composite_loss(
                            prediction,
                            target,
                            include_geodesic=include_geodesic,
                            include_smoothness=include_smoothness,
                            include_sparsity=include_sparsity,
                        )
                        scaled_loss = loss / CFG.accumulation_steps

                    if training:
                        scaler.scale(scaled_loss).backward()
                        should_update = (
                            (step + 1) % CFG.accumulation_steps == 0
                            or step + 1 == len(data_loader)
                        )
                        if should_update:
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(
                                model.parameters(), 1.0
                            )
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)

                    loss_sum += float(loss.detach().cpu()) * len(target)
                    sample_count += len(target)

            epoch_record[phase] = loss_sum / max(sample_count, 1)

        scheduler.step(epoch_record["validation"])
        history.append(epoch_record)
        print(epoch_record)

        if epoch_record["validation"] < best_validation - 1e-7:
            best_validation = epoch_record["validation"]
            stale_epochs = 0
            save_training_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_validation,
                stale_epochs,
            )
        else:
            stale_epochs += 1

        if stale_epochs >= CFG.early_stopping:
            break

    if not checkpoint_path.exists():
        raise RuntimeError("No training checkpoint was saved")

    print_deep_assessment(f"Training Complete for Checkpoint: {checkpoint_path.name}")

    best_checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model"])
    model.eval()
    return model, pd.DataFrame(history)


# %% 28 — MODEL EVALUATION
@torch.no_grad()
def evaluate_model(model, input_kind, dataset):
    model.eval()
    rows = []
    physics_times = []
    graph_times = []
    network_times = []
    peak_gpu_mb = 0.0

    for index in range(len(dataset)):
        item = dataset[index]
        eeg_scaled = item["eeg"].numpy()
        target = item["source"].numpy()
        centers = item["centers"].numpy()

        if input_kind == "eeg":
            model_input = torch.from_numpy(
                eeg_scaled[None]
            ).to(DEVICE)
            edge_indices = [item["edge_index"]]
            edge_attributes = [item["edge_attr"]]
            physics_times.append(0.0)
            graph_times.append(0.0)
        else:
            physical_eeg = eeg_scaled * SCALES[0]

            start = time.perf_counter()
            initial_source = dataset.inverse_operator @ physical_eeg
            initial_source = np.clip(
                initial_source / SCALES[1], -10.0, 10.0
            ).astype(np.float32)
            physics_times.append(
                (time.perf_counter() - start) * 1000.0
            )

            start = time.perf_counter()
            edge_index, edge_attr = dataset.graph_builder(
                initial_source
            )
            graph_times.append(
                (time.perf_counter() - start) * 1000.0
            )

            model_input = torch.from_numpy(
                initial_source[None]
            ).to(DEVICE)
            edge_indices = [edge_index]
            edge_attributes = [edge_attr]

        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start = time.perf_counter()
        prediction = model(
            model_input,
            edge_indices,
            edge_attributes,
        )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
            peak_gpu_mb = max(
                peak_gpu_mb,
                torch.cuda.max_memory_allocated() / 1024**2,
            )

        network_times.append(
            (time.perf_counter() - start) * 1000.0
        )
        prediction_numpy = prediction[0].float().cpu().numpy()
        rows.append(
            compute_sample_metrics(
                target,
                prediction_numpy,
                centers,
            )
        )

    timing = {
        "physics_ms": float(np.mean(physics_times)),
        "graph_ms": float(np.mean(graph_times)),
        "network_ms": float(np.mean(network_times)),
        "total_ms": float(
            np.mean(physics_times)
            + np.mean(graph_times)
            + np.mean(network_times)
        ),
        "peak_gpu_mb": float(peak_gpu_mb),
    }
    return pd.DataFrame(rows), timing


# %% 29 — ABLATION REGISTRY
MODEL_SPECS = {
    "fair_convdip_cnn": {
        "factory": FairConvDipCNN,
        "input_kind": "eeg",
        "dynamic_graph": False,
        "geodesic": True,
        "smoothness": True,
        "sparsity": True,
    },
    "static_full_physics_gat": {
        "factory": FullPhysicsGAT,
        "input_kind": "initial",
        "dynamic_graph": False,
        "geodesic": True,
        "smoothness": True,
        "sparsity": True,
    },
    "dynamic_full_physics_gat": {
        "factory": FullPhysicsGAT,
        "input_kind": "initial",
        "dynamic_graph": True,
        "geodesic": True,
        "smoothness": True,
        "sparsity": True,
    },
    "threshold_physics_gat": {
        "factory": ThresholdPhysicsGAT,
        "input_kind": "initial",
        "dynamic_graph": True,
        "geodesic": True,
        "smoothness": True,
        "sparsity": True,
    },
    "sagpool_physics_gat": {
        "factory": SAGPoolPhysicsGAT,
        "input_kind": "initial",
        "dynamic_graph": True,
        "geodesic": True,
        "smoothness": True,
        "sparsity": True,
    },
    "gat_without_geodesic": {
        "factory": FullPhysicsGAT,
        "input_kind": "initial",
        "dynamic_graph": True,
        "geodesic": False,
        "smoothness": True,
        "sparsity": True,
    },
    "gat_without_smoothness": {
        "factory": FullPhysicsGAT,
        "input_kind": "initial",
        "dynamic_graph": True,
        "geodesic": True,
        "smoothness": False,
        "sparsity": True,
    },
    "gat_without_sparsity": {
        "factory": FullPhysicsGAT,
        "input_kind": "initial",
        "dynamic_graph": True,
        "geodesic": True,
        "smoothness": True,
        "sparsity": False,
    },
}

if ROI_INDEX is not None:
    MODEL_SPECS["atlas_physics_gat"] = {
        "factory": lambda: AtlasPhysicsGAT(ROI_INDEX),
        "input_kind": "initial",
        "dynamic_graph": False,
        "geodesic": True,
        "smoothness": True,
        "sparsity": True,
    }

if CFG.smoke_test:
    _smoke_model_names = (
        "fair_convdip_cnn",
        "dynamic_full_physics_gat",
        "threshold_physics_gat",
        "sagpool_physics_gat",
    )
    MODEL_SPECS = {
        name: MODEL_SPECS[name]
        for name in _smoke_model_names
    }


# %% 30 — TRAIN, ID/OOD, SENSOR MASKING, AND CLASSICAL BASELINE
TEST_CONDITIONS = {
    "id": {},
    "ood_snr": {"condition": "ood_snr"},
    "ood_sources": {"condition": "ood_sources"},
    "ood_extent": {"condition": "ood_extent"},
    "algebraic_leadfield_perturbation": {
        "condition": "id",
        "lead_field": PERTURBED_LEAD_FIELD,
        "inverse_operator": PERTURBED_OPERATOR,
    },
}

ALL_SUMMARY_ROWS = []
METRIC_STORE = {}

print_deep_assessment("Entering Execution Loop")

for seed in CFG.seeds:
    seed_all(seed)

    for model_name, specification in MODEL_SPECS.items():
        dynamic_graph = specification["dynamic_graph"]

        train_dataset = make_dataset(
            "train",
            condition="id",
            dynamic_graph=dynamic_graph,
        )
        validation_dataset = make_dataset(
            "validation",
            condition="id",
            dynamic_graph=dynamic_graph,
        )

        model = specification["factory"]()
        checkpoint_path = OUT / f"{model_name}_seed_{seed}.pt"
        start = time.time()

        model, history = train_model(
            model=model,
            input_kind=specification["input_kind"],
            train_loader=make_loader(train_dataset, shuffle=True),
            validation_loader=make_loader(
                validation_dataset, shuffle=False
            ),
            checkpoint_path=checkpoint_path,
            include_geodesic=specification["geodesic"],
            include_smoothness=specification["smoothness"],
            include_sparsity=specification["sparsity"],
        )

        training_minutes = (time.time() - start) / 60.0
        history.to_csv(
            OUT / f"history_{model_name}_seed_{seed}.csv",
            index=False,
        )

        for condition_name, condition_options in TEST_CONDITIONS.items():
            test_dataset = make_dataset(
                "test",
                condition=condition_options.get("condition", "id"),
                dynamic_graph=dynamic_graph,
                lead_field=condition_options.get(
                    "lead_field", LEAD_FIELD
                ),
                inverse_operator=condition_options.get(
                    "inverse_operator", TRAIN_OPERATOR
                ),
            )

            metrics, timing = evaluate_model(
                model,
                specification["input_kind"],
                test_dataset,
            )
            METRIC_STORE[(model_name, condition_name, seed)] = metrics
            metrics.to_csv(
                OUT
                / f"metrics_{model_name}_{condition_name}_seed_{seed}.csv",
                index=False,
            )

            summary_row = metrics.mean(
                numeric_only=True
            ).to_dict()
            summary_row.update(
                {
                    "model": model_name,
                    "condition": condition_name,
                    "seed": seed,
                    "parameters": parameter_count(model),
                    "training_minutes": training_minutes,
                    **timing,
                }
            )
            ALL_SUMMARY_ROWS.append(summary_row)

        base_test_dataset = make_dataset(
            "test",
            condition="id",
            dynamic_graph=dynamic_graph,
        )
        for sensor_fraction in (0.10, 0.25, 0.40):
            condition_name = f"sensor_mask_{sensor_fraction:.2f}"
            masked_dataset = SensorMaskDataset(
                base_test_dataset,
                sensor_fraction,
            )
            metrics, timing = evaluate_model(
                model,
                specification["input_kind"],
                masked_dataset,
            )
            METRIC_STORE[(model_name, condition_name, seed)] = metrics
            metrics.to_csv(
                OUT
                / f"metrics_{model_name}_{condition_name}_seed_{seed}.csv",
                index=False,
            )

            summary_row = metrics.mean(
                numeric_only=True
            ).to_dict()
            summary_row.update(
                {
                    "model": model_name,
                    "condition": condition_name,
                    "seed": seed,
                    "parameters": parameter_count(model),
                    "training_minutes": training_minutes,
                    **timing,
                }
            )
            ALL_SUMMARY_ROWS.append(summary_row)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    classical_dataset = make_dataset(
        "test",
        condition="id",
        dynamic_graph=False,
    )
    classical_metrics, classical_timing = evaluate_tikhonov(
        classical_dataset
    )
    METRIC_STORE[("tikhonov", "id", seed)] = classical_metrics
    classical_metrics.to_csv(
        OUT / f"metrics_tikhonov_id_seed_{seed}.csv",
        index=False,
    )

    classical_row = classical_metrics.mean(
        numeric_only=True
    ).to_dict()
    classical_row.update(
        {
            "model": "tikhonov",
            "condition": "id",
            "seed": seed,
            "parameters": 0,
            "training_minutes": 0.0,
            **classical_timing,
        }
    )
    ALL_SUMMARY_ROWS.append(classical_row)

SUMMARY = pd.DataFrame(ALL_SUMMARY_ROWS)
SUMMARY.to_csv(OUT / "benchmark_summary.csv", index=False)


# %% 31 — PAIRED STATISTICS
def paired_statistical_test(first_dataframe, second_dataframe, metric):
    first = first_dataframe[metric].to_numpy(dtype=float)
    second = second_dataframe[metric].to_numpy(dtype=float)
    valid = np.isfinite(first) & np.isfinite(second)
    difference = first[valid] - second[valid]

    if len(difference) == 0:
        return None

    rng = np.random.default_rng(123)
    bootstrap_means = np.asarray(
        [
            rng.choice(
                difference,
                len(difference),
                replace=True,
            ).mean()
            for _ in range(3000)
        ]
    )

    if np.allclose(difference, 0.0):
        statistic, p_value = 0.0, 1.0
    else:
        statistic, p_value = wilcoxon(difference)

    return {
        "metric": metric,
        "n": int(len(difference)),
        "mean_difference": float(difference.mean()),
        "ci_low": float(np.quantile(bootstrap_means, 0.025)),
        "ci_high": float(np.quantile(bootstrap_means, 0.975)),
        "wilcoxon_statistic": float(statistic),
        "p_value": float(p_value),
    }


STATISTICS_ROWS = []
for seed in CFG.seeds:
    for condition_name in SUMMARY["condition"].unique():
        proposed = METRIC_STORE.get(
            ("dynamic_full_physics_gat", condition_name, seed)
        )
        baseline = METRIC_STORE.get(
            ("fair_convdip_cnn", condition_name, seed)
        )
        if proposed is None or baseline is None:
            continue

        for metric_name in (
            "waveform_mse_sign_invariant",
            "map_mse",
            "auc",
            "f1",
            "dle_mm",
        ):
            result = paired_statistical_test(
                proposed,
                baseline,
                metric_name,
            )
            if result is not None:
                result.update(
                    {
                        "seed": seed,
                        "condition": condition_name,
                    }
                )
                STATISTICS_ROWS.append(result)

STATISTICS = pd.DataFrame(STATISTICS_ROWS)
if not STATISTICS.empty:
    STATISTICS["p_holm"] = multipletests(
        STATISTICS["p_value"].to_numpy(),
        method="holm",
    )[1]
STATISTICS.to_csv(OUT / "paired_statistics.csv", index=False)


# %% 32 — OPTIONAL REAL EEG ANALYSIS
def prepare_real_evokeds():
    events = mne.find_events(
        RAW,
        stim_channel="STI 014",
        shortest_event=1,
        verbose=False,
    )
    event_id = {
        "Auditory_Left": 1,
        "Auditory_Right": 2,
        "Visual_Left": 3,
        "Visual_Right": 4,
    }
    present_event_id = {
        name: code
        for name, code in event_id.items()
        if np.any(events[:, 2] == code)
    }
    if not present_event_id:
        raise RuntimeError("No requested real EEG events were found")

    eeg_picks = mne.pick_types(
        RAW.info,
        meg=False,
        eeg=True,
        eog=False,
        stim=False,
        exclude="bads",
    )
    epochs = mne.Epochs(
        RAW,
        events,
        event_id=present_event_id,
        tmin=-0.20,
        tmax=0.50,
        baseline=(-0.20, 0.0),
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

    forward_channels = list(FWD_FIXED["info"]["ch_names"])
    missing_channels = [
        channel
        for channel in forward_channels
        if channel not in epochs.ch_names
    ]
    if missing_channels:
        raise RuntimeError(
            f"Real EEG is missing forward channels: {missing_channels}"
        )
    epochs.pick(forward_channels)
    epochs.reorder_channels(forward_channels)

    evokeds = {
        name: epochs[name].average().crop(0.0, CFG.duration_s)
        for name in present_event_id
        if len(epochs[name]) > 0
    }
    return epochs, evokeds


def predict_real_graph(model, evoked):
    eeg = np.asarray(evoked.data, dtype=np.float32)
    if eeg.shape != (N_CHANNELS, N_TIMES):
        raise ValueError(
            f"Real evoked must have shape ({N_CHANNELS},{N_TIMES}), "
            f"got {eeg.shape}"
        )

    initial_source = TRAIN_OPERATOR @ eeg
    initial_source = np.clip(
        initial_source / SCALES[1], -10.0, 10.0
    ).astype(np.float32)
    edge_index, edge_attr = DYNAMIC_GRAPH_BUILDER(initial_source)

    model.eval()
    with torch.no_grad():
        prediction = model(
            torch.from_numpy(initial_source[None]).to(DEVICE),
            [edge_index],
            [edge_attr],
        )[0]
    prediction = prediction.float().cpu().numpy() * SCALES[2]

    return mne.SourceEstimate(
        prediction,
        vertices=[
            FWD_FIXED["src"][0]["vertno"],
            FWD_FIXED["src"][1]["vertno"],
        ],
        tmin=float(evoked.times[0]),
        tstep=1.0 / float(evoked.info["sfreq"]),
        subject="sample",
    )


def run_real_eeg_analysis(model):
    epochs, evokeds = prepare_real_evokeds()
    noise_covariance = mne.compute_covariance(
        epochs,
        tmin=-0.20,
        tmax=0.0,
        method=["shrunk", "empirical"],
        rank=None,
        verbose=False,
    )
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        epochs.info,
        FWD_FREE,
        noise_covariance,
        loose=0.2,
        depth=0.8,
        rank=None,
        verbose=False,
    )

    agreement_rows = []
    for condition_name, evoked in evokeds.items():
        graph_stc = predict_real_graph(model, evoked)
        graph_stc.save(
            str(OUT / f"real_graph_{condition_name}"),
            overwrite=True,
        )
        graph_map = np.max(np.abs(graph_stc.data), axis=1)
        graph_map /= graph_map.max() + 1e-12

        for method in ("MNE", "dSPM", "sLORETA", "eLORETA"):
            try:
                classical_stc = mne.minimum_norm.apply_inverse(
                    evoked,
                    inverse_operator,
                    lambda2=1.0 / 9.0,
                    method=method,
                    pick_ori=None,
                    verbose=False,
                )
            except Exception as error:
                print(
                    "Skipping",
                    method,
                    condition_name,
                    str(error),
                )
                continue

            classical_stc.save(
                str(OUT / f"real_{method}_{condition_name}"),
                overwrite=True,
            )
            classical_map = np.max(
                np.abs(classical_stc.data), axis=1
            )
            classical_map /= classical_map.max() + 1e-12

            graph_peak = int(np.argmax(graph_map))
            classical_peak = int(np.argmax(classical_map))
            agreement_rows.append(
                {
                    "condition": condition_name,
                    "reference": method,
                    "map_mse_agreement": float(
                        np.mean((graph_map - classical_map) ** 2)
                    ),
                    "cosine_agreement": float(
                        np.sum(graph_map * classical_map)
                        / (
                            np.linalg.norm(graph_map)
                            * np.linalg.norm(classical_map)
                            + 1e-12
                        )
                    ),
                    "peak_geodesic_disagreement_mm": float(
                        GEODESIC_MM[graph_peak, classical_peak]
                    ),
                    "interpretation": "agreement_not_ground_truth",
                }
            )

    pd.DataFrame(agreement_rows).to_csv(
        OUT / "real_eeg_agreement.csv",
        index=False,
    )


if CFG.run_real_eeg:
    real_checkpoint = (
        OUT / f"dynamic_full_physics_gat_seed_{CFG.seeds[0]}.pt"
    )
    if not real_checkpoint.exists():
        raise FileNotFoundError(
            "Train dynamic_full_physics_gat before real EEG analysis"
        )
    real_model = FullPhysicsGAT().to(DEVICE)
    real_model.load_state_dict(
        torch.load(
            real_checkpoint,
            map_location=DEVICE,
            weights_only=False,
        )["model"]
    )
    real_model.eval()
    run_real_eeg_analysis(real_model)


# %% 33 — CLINICAL/iEEG/STIMULATION AND EXPERT INTERFACES
def nearest_source_vertices(reference_xyz_mm):
    reference_xyz_mm = np.asarray(reference_xyz_mm, dtype=float)
    if reference_xyz_mm.ndim != 2 or reference_xyz_mm.shape[1] != 3:
        raise ValueError(
            "reference_xyz_mm must have shape [n,3] in the same MRI space"
        )
    return np.argmin(
        np.linalg.norm(
            COORDINATES_MM[None, :, :]
            - reference_xyz_mm[:, None, :],
            axis=2,
        ),
        axis=1,
    )


def load_bids_electrodes_tsv(
    electrodes_tsv,
    coordinate_scale_to_mm=1.0,
    onset_column=None,
):
    dataframe = pd.read_csv(electrodes_tsv, sep="	")
    required = {"x", "y", "z"}
    if not required.issubset(dataframe.columns):
        raise ValueError("electrodes TSV needs x, y, and z columns")
    if onset_column is not None:
        if onset_column not in dataframe.columns:
            raise ValueError(f"Missing onset column: {onset_column}")
        dataframe = dataframe[dataframe[onset_column].astype(bool)]
    return (
        dataframe[["x", "y", "z"]].to_numpy(dtype=float)
        * coordinate_scale_to_mm
    )


def evaluate_clinical_reference(predicted_map, reference_xyz_mm):
    references = nearest_source_vertices(reference_xyz_mm)
    predicted_centers = extract_separated_peaks(predicted_map)
    normalized_map = np.abs(predicted_map) / (
        np.max(np.abs(predicted_map)) + 1e-12
    )
    ranks = np.argsort(np.argsort(-normalized_map))[references] + 1

    return {
        "minimum_geodesic_distance_mm": float(
            np.min(
                GEODESIC_MM[
                    np.ix_(references, predicted_centers)
                ]
            )
        ),
        "best_reference_rank": int(np.min(ranks)),
        "mean_reference_normalized_activity": float(
            np.mean(normalized_map[references])
        ),
    }


def create_blinded_expert_form(
    case_ids,
    algorithm_names,
    output_path,
    seed=123,
):
    rng = np.random.default_rng(seed)
    rows = []
    for case_id in case_ids:
        hidden_order = rng.permutation(algorithm_names)
        for display_order, algorithm in enumerate(hidden_order):
            rows.append(
                {
                    "case_id": case_id,
                    "display_order": display_order,
                    "hidden_algorithm": algorithm,
                    "anatomical_plausibility_0_to_4": np.nan,
                    "clinical_agreement_0_to_4": np.nan,
                    "comments": "",
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


# %% 34 — OUTPUTS, MANIFEST, AND FINAL SANITY CHECKS
OUTPUT_METRICS = (
    "map_mse",
    "auc",
    "f1",
    "dle_mm",
    "waveform_mse_sign_invariant",
    "total_ms",
    "peak_gpu_mb",
)

for metric_name in OUTPUT_METRICS:
    if metric_name not in SUMMARY.columns:
        continue

    plot_data = SUMMARY[
        np.isfinite(
            SUMMARY[metric_name].to_numpy(dtype=float)
        )
    ].copy()
    if plot_data.empty:
        continue

    plt.figure(figsize=(15, 6))
    sns.barplot(
        data=plot_data,
        x="condition",
        y=metric_name,
        hue="model",
        errorbar="sd",
    )
    plt.title(metric_name)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(
        OUT / f"comparison_{metric_name}.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close()


def final_sanity_checks():
    if SUMMARY.empty:
        raise RuntimeError("Benchmark summary is empty")

    required_columns = {
        "model",
        "condition",
        "seed",
        "map_mse",
        "dle_mm",
    }
    missing = required_columns.difference(SUMMARY.columns)
    if missing:
        raise RuntimeError(
            f"Summary is missing columns: {sorted(missing)}"
        )

    if not np.isfinite(
        SUMMARY["map_mse"].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("Non-finite map MSE values were found")
    if not np.isfinite(
        SUMMARY["dle_mm"].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("Non-finite DLE values were found")

    checkpoint_files = sorted(OUT.glob("*_seed_*.pt"))
    if not checkpoint_files:
        raise RuntimeError("No model checkpoint was saved")

    print("FINAL SANITY CHECKS PASSED")


final_sanity_checks()
print_deep_assessment("Pipeline Completed Successfully")

with open(
    OUT / "final_configuration.json",
    "w",
    encoding="utf-8",
) as handle:
    json.dump(asdict(CFG), handle, ensure_ascii=False, indent=2)

manifest = {
    "output_directory": str(OUT),
    "device": str(DEVICE),
    "n_channels": int(N_CHANNELS),
    "n_vertices": int(N_VERTICES),
    "n_times": int(N_TIMES),
    "smoke_test": bool(CFG.smoke_test),
    "models": list(MODEL_SPECS.keys()) + ["tikhonov"],
    "conditions": list(TEST_CONDITIONS.keys())
    + ["sensor_mask_0.10", "sensor_mask_0.25", "sensor_mask_0.40"],
    "scientific_warnings": [
        "FairConvDipCNN is not official ESINet ConvDip.",
        "Algebraic lead-field perturbation is a controlled surrogate.",
        "Real MNE inverse methods are agreement references, not ground truth.",
        "AAL needs a validated subject-to-MNI vertex mapping.",
    ],
}
with open(OUT / "manifest.json", "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, ensure_ascii=False, indent=2)

SUMMARY.to_csv(OUT / "benchmark_summary.csv", index=False)
if not STATISTICS.empty:
    STATISTICS.to_csv(OUT / "paired_statistics.csv", index=False)

print(SUMMARY.round(4).to_string(index=False))
print("Results saved in:", OUT)
print("PART 3 COMPLETED SUCCESSFULLY")
print("FULL PIPELINE COMPLETED SUCCESSFULLY")
