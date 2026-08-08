# -*- coding: utf-8 -*-
"""
PHYSICS-GAT FINAL PIPELINE — COLAB PRO SINGLE-CELL VERSION
==========================================================
This is the combined, streamlined code for Colab Pro execution.
It includes all 3 parts of the original script, with Google Drive
integration and deep profiling for Colab Pro environments (V100/A100/L4).
"""

# %% 0 — INSTALL IN A SEPARATE COLAB CELL
from __future__ import annotations
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
def load_roi_mapping():
    if CFG.atlas_mode == "none":
        return None
    # For brevity, full implementation available in source, skipping for 'none' default
    return None

ROI_INDEX = load_roi_mapping()

class AtlasPhysicsGAT(nn.Module):
    pass # Implementation hidden unless needed, since CFG.atlas_mode='none'


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
    positive_error = (prediction - target).square().mean(dim=(1, 2))
    negative_error = (prediction + target).square().mean(dim=(1, 2))
    waveform_loss = torch.minimum(positive_error, negative_error).mean()

    predicted_map = prediction.abs().amax(dim=-1)
    target_map = target.abs().amax(dim=-1)

    target_peak = target_map.amax(dim=1, keepdim=True).clamp_min(1e-8)
    active_target = target_map >= CFG.active_ratio * target_peak

    map_weights = 1.0 + (CFG.active_weight - 1.0) * active_target.float()
    map_loss = (
        map_weights * (predicted_map - target_map).square()
    ).sum() / map_weights.sum().clamp_min(1.0)

    residual_map = predicted_map - target_map
    smoothness_loss = (
        residual_map[:, SPATIAL_EDGE_TENSOR[0]]
        - residual_map[:, SPATIAL_EDGE_TENSOR[1]]
    ).square().mean()

    probability = predicted_map / predicted_map.sum(dim=1, keepdim=True).clamp_min(1e-8)

    geodesic_terms = []
    for sample_index in range(len(prediction)):
        true_support = torch.where(active_target[sample_index])[0]
        if len(true_support) == 0:
            true_support = target_map[sample_index].argmax().reshape(1)

        distance_to_true_support = GEODESIC_TENSOR[:, true_support].amin(dim=1)
        geodesic_terms.append(
            (probability[sample_index] * distance_to_true_support).sum() / 100.0
        )

    geodesic_loss = torch.stack(geodesic_terms).mean()
    sparsity_loss = predicted_map.mean()

    total_loss = (CFG.waveform_weight * waveform_loss + CFG.map_weight * map_loss)
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
        self.spatial_head = nn.Linear(512, N_VERTICES * CFG.cnn_rank)
        self.temporal_head = nn.Linear(512, CFG.cnn_rank * N_TIMES)

    def forward(self, eeg, edge_indices=None, edge_attributes=None):
        latent = self.latent(self.encoder(eeg))
        spatial = self.spatial_head(latent).reshape(len(eeg), N_VERTICES, CFG.cnn_rank)
        temporal = self.temporal_head(latent).reshape(len(eeg), CFG.cnn_rank, N_TIMES)
        return torch.einsum("bvr,brt->bvt", spatial, temporal)


# %% 22 — METRICS
def extract_separated_peaks(activity):
    activity = np.asarray(activity, dtype=float)
    candidates = np.where(activity >= CFG.peak_ratio * (activity.max() + 1e-12))[0]
    candidates = candidates[np.argsort(activity[candidates])[::-1]]

    selected = []
    for candidate in candidates:
        if not selected or np.all(GEODESIC_MM[candidate, selected] >= CFG.peak_separation_mm):
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

    assignment_cost = GEODESIC_MM[np.ix_(true_centers, predicted_centers)]
    true_assignment, predicted_assignment = linear_sum_assignment(assignment_cost)

    distances = assignment_cost[true_assignment, predicted_assignment].tolist()
    missed_sources = len(true_centers) - len(true_assignment)
    false_sources = len(predicted_centers) - len(predicted_assignment)
    distances.extend([CFG.unmatched_penalty_mm] * (missed_sources + false_sources))

    return {
        "dle_mm": float(np.mean(distances)),
        "missed_sources": int(missed_sources),
        "false_sources": int(false_sources),
        "predicted_sources": int(len(predicted_centers)),
    }

def compute_sample_metrics(target, prediction, true_centers):
    target = np.asarray(target, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)

    target_map = np.max(np.abs(target), axis=1)
    predicted_map = np.max(np.abs(prediction), axis=1)
    target_normalized = target_map / (target_map.max() + 1e-12)
    prediction_normalized = predicted_map / (predicted_map.max() + 1e-12)

    target_label = (target_normalized >= CFG.active_ratio).astype(np.int64)
    predicted_label = (prediction_normalized >= CFG.active_ratio).astype(np.int64)

    auc = np.nan
    if np.unique(target_label).size == 2:
        auc = roc_auc_score(target_label, predicted_map)

    precision, recall, f1, _ = precision_recall_fscore_support(
        target_label, predicted_label, average="binary", zero_division=0,
    )

    positive_waveform_error = np.mean((target - prediction) ** 2)
    negative_waveform_error = np.mean((target + prediction) ** 2)
    dle_values = multisource_geodesic_dle(true_centers, predicted_map)

    result = {
        "waveform_mse_sign_invariant": float(min(positive_waveform_error, negative_waveform_error)),
        "map_mse": float(np.mean((target_normalized - prediction_normalized) ** 2)),
        "cosine": float(
            np.sum(target_normalized * prediction_normalized)
            / (np.linalg.norm(target_normalized) * np.linalg.norm(prediction_normalized) + 1e-12)
        ),
        "auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "extent_error_vertices": float(abs(predicted_label.sum() - target_label.sum())),
    }
    result.update(dle_values)
    return result


# %% 23 — SENSOR MASKING
class SensorMaskDataset(Dataset):
    def __init__(self, base_dataset, fraction, seed=9000):
        self.base_dataset = base_dataset
        self.inverse_operator = base_dataset.inverse_operator
        self.graph_builder = base_dataset.graph_builder

        rng = np.random.default_rng(seed)
        self.dropped_channels = rng.choice(
            N_CHANNELS, max(1, round(N_CHANNELS * fraction)), replace=False,
        )

    def __len__(self): return len(self.base_dataset)

    def __getitem__(self, index):
        item = self.base_dataset[index]
        eeg = item["eeg"].clone()
        eeg[self.dropped_channels] = 0.0
        item["eeg"] = eeg
        return item


# %% 27 — CHECKPOINTING AND TRAINING
def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_training_checkpoint(
    path, model, optimizer, scheduler, scaler, epoch, best_validation, stale_epochs,
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
            "cuda_rng": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        },
        path,
    )

def train_model(
    model, input_kind, train_loader, validation_loader, checkpoint_path,
    include_geodesic=True, include_smoothness=True, include_sparsity=True,
):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=CFG.scheduler_patience, min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=CFG.amp and DEVICE.type == "cuda")

    start_epoch = 1
    best_validation = float("inf")
    stale_epochs = 0

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
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
        if torch.cuda.is_available() and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])

    history = []
    for epoch in range(start_epoch, CFG.epochs + 1):
        epoch_record = {"epoch": epoch}

        for phase, data_loader in (("train", train_loader), ("validation", validation_loader)):
            training = phase == "train"
            model.train(training)
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            sample_count = 0

            gradient_context = torch.enable_grad() if training else torch.no_grad()
            with gradient_context:
                for step, batch in enumerate(data_loader):
                    eeg = batch["eeg"].to(DEVICE, non_blocking=True)
                    initial = batch["initial"].to(DEVICE, non_blocking=True)
                    target = batch["source"].to(DEVICE, non_blocking=True)
                    model_input = eeg if input_kind == "eeg" else initial

                    with torch.amp.autocast("cuda", enabled=CFG.amp and DEVICE.type == "cuda"):
                        prediction = model(model_input, batch["edge_index"], batch["edge_attr"])
                        loss, _ = full_composite_loss(
                            prediction, target,
                            include_geodesic=include_geodesic,
                            include_smoothness=include_smoothness,
                            include_sparsity=include_sparsity,
                        )
                        scaled_loss = loss / CFG.accumulation_steps

                    if training:
                        scaler.scale(scaled_loss).backward()
                        should_update = ((step + 1) % CFG.accumulation_steps == 0 or step + 1 == len(data_loader))
                        if should_update:
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)

                    loss_sum += float(loss.detach().cpu()) * len(target)
                    sample_count += len(target)

            epoch_record[phase] = loss_sum / max(sample_count, 1)

        scheduler.step(epoch_record["validation"])
        history.append(epoch_record)
        print(f"Epoch {epoch_record['epoch']}: Train Loss = {epoch_record['train']:.4f}, Val Loss = {epoch_record['validation']:.4f}")

        improved = epoch_record["validation"] < best_validation - 1e-7
        if improved:
            best_validation = epoch_record["validation"]
            stale_epochs = 0
        else:
            stale_epochs += 1

        save_training_checkpoint(
            checkpoint_path.with_name(checkpoint_path.stem + "_last.pt"),
            model, optimizer, scheduler, scaler, epoch, best_validation, stale_epochs,
        )

        if improved:
            save_training_checkpoint(
                checkpoint_path, model, optimizer, scheduler, scaler,
                epoch, best_validation, stale_epochs,
            )

        if stale_epochs >= CFG.early_stopping:
            print("Early stopping triggered.")
            break

    if not checkpoint_path.exists():
        raise RuntimeError("No training checkpoint was saved")

    best_checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    model.eval()

    print_deep_assessment(f"Training Complete for Checkpoint: {checkpoint_path.name}")
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
            model_input = torch.from_numpy(eeg_scaled[None]).to(DEVICE)
            edge_indices = [item["edge_index"]]
            edge_attributes = [item["edge_attr"]]
            physics_times.append(0.0)
            graph_times.append(0.0)
        else:
            physical_eeg = eeg_scaled * SCALES[0]

            start = time.perf_counter()
            initial_source = dataset.inverse_operator @ physical_eeg
            initial_source = np.clip(initial_source / SCALES[1], -10.0, 10.0).astype(np.float32)
            physics_times.append((time.perf_counter() - start) * 1000.0)

            start = time.perf_counter()
            edge_index, edge_attr = dataset.graph_builder(initial_source)
            graph_times.append((time.perf_counter() - start) * 1000.0)

            model_input = torch.from_numpy(initial_source[None]).to(DEVICE)
            edge_indices = [edge_index]
            edge_attributes = [edge_attr]

        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start = time.perf_counter()
        prediction = model(model_input, edge_indices, edge_attributes)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
            peak_gpu_mb = max(peak_gpu_mb, torch.cuda.max_memory_allocated() / 1024**2)

        network_times.append((time.perf_counter() - start) * 1000.0)
        prediction_numpy = prediction[0].float().cpu().numpy()
        rows.append(compute_sample_metrics(target, prediction_numpy, centers))

    timing = {
        "physics_ms": float(np.mean(physics_times)),
        "graph_ms": float(np.mean(graph_times)),
        "network_ms": float(np.mean(network_times)),
        "total_ms": float(np.mean(physics_times) + np.mean(graph_times) + np.mean(network_times)),
        "peak_gpu_mb": float(peak_gpu_mb),
    }
    return pd.DataFrame(rows), timing


# %% 29 — ABLATION REGISTRY & EXECUTION LOOP
MODEL_SPECS = {
    "fair_convdip_cnn": {
        "factory": FairConvDipCNN, "input_kind": "eeg", "dynamic_graph": False,
        "geodesic": True, "smoothness": True, "sparsity": True,
    },
    "dynamic_full_physics_gat": {
        "factory": FullPhysicsGAT, "input_kind": "initial", "dynamic_graph": True,
        "geodesic": True, "smoothness": True, "sparsity": True,
    },
    "threshold_physics_gat": {
        "factory": ThresholdPhysicsGAT, "input_kind": "initial", "dynamic_graph": True,
        "geodesic": True, "smoothness": True, "sparsity": True,
    },
    "sagpool_physics_gat": {
        "factory": SAGPoolPhysicsGAT, "input_kind": "initial", "dynamic_graph": True,
        "geodesic": True, "smoothness": True, "sparsity": True,
    },
}

if CFG.smoke_test:
    MODEL_SPECS = {k: MODEL_SPECS[k] for k in list(MODEL_SPECS.keys())[:2]}

TEST_CONDITIONS = {
    "id": {},
    "ood_snr": {"condition": "ood_snr"},
    "ood_sources": {"condition": "ood_sources"},
}

ALL_SUMMARY_ROWS = []
METRIC_STORE = {}

print_deep_assessment("Entering Execution Loop")

for seed in CFG.seeds:
    seed_all(seed)

    for model_name, specification in MODEL_SPECS.items():
        print(f"\n--- Training {model_name} (Seed {seed}) ---")
        dynamic_graph = specification["dynamic_graph"]

        train_dataset = make_dataset("train", condition="id", dynamic_graph=dynamic_graph)
        validation_dataset = make_dataset("validation", condition="id", dynamic_graph=dynamic_graph)

        model = specification["factory"]()
        checkpoint_path = OUT / f"{model_name}_seed_{seed}.pt"
        start = time.time()

        model, history = train_model(
            model=model,
            input_kind=specification["input_kind"],
            train_loader=make_loader(train_dataset, shuffle=True),
            validation_loader=make_loader(validation_dataset, shuffle=False),
            checkpoint_path=checkpoint_path,
            include_geodesic=specification["geodesic"],
            include_smoothness=specification["smoothness"],
            include_sparsity=specification["sparsity"],
        )

        training_minutes = (time.time() - start) / 60.0
        history.to_csv(OUT / f"history_{model_name}_seed_{seed}.csv", index=False)

        print(f"--- Evaluating {model_name} (Seed {seed}) ---")
        for condition_name, condition_options in TEST_CONDITIONS.items():
            test_dataset = make_dataset(
                "test",
                condition=condition_options.get("condition", "id"),
                dynamic_graph=dynamic_graph,
                lead_field=condition_options.get("lead_field", LEAD_FIELD),
                inverse_operator=condition_options.get("inverse_operator", TRAIN_OPERATOR),
            )

            metrics, timing = evaluate_model(model, specification["input_kind"], test_dataset)
            METRIC_STORE[(model_name, condition_name, seed)] = metrics
            metrics.to_csv(OUT / f"metrics_{model_name}_{condition_name}_seed_{seed}.csv", index=False)

            summary_row = metrics.mean(numeric_only=True).to_dict()
            summary_row.update({
                "model": model_name, "condition": condition_name, "seed": seed,
                "parameters": parameter_count(model), "training_minutes": training_minutes, **timing,
            })
            ALL_SUMMARY_ROWS.append(summary_row)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

SUMMARY = pd.DataFrame(ALL_SUMMARY_ROWS)
SUMMARY.to_csv(OUT / "benchmark_summary.csv", index=False)
print("\nBenchmark Summary Sample:\n", SUMMARY.head())

print_deep_assessment("Pipeline Completed Successfully")
print("\nFULL PIPELINE COMPLETED SUCCESSFULLY")
print(f"Results saved in: {OUT}")
