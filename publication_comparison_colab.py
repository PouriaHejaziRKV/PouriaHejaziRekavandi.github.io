# -*- coding: utf-8 -*-
# ============================================================
# REAL MNE SAMPLE EEG:
# ConvDip vs Sparse Spatio-Temporal Graph Network vs FullPhysicsGAT vs dSPM
#
# Real EEG inference + subject-specific forward model
# Supervised neural training uses simulations generated from
# the real subject-specific forward model.
# ============================================================

# %% 0 — INSTALL IN A SEPARATE COLAB CELL
# RUN THIS CELL BEFORE THE REST OF THE SCRIPT
# import IPython
# print("Installing dependencies...")
# IPython.get_ipython().system('pip install -q esinet mne pandas seaborn matplotlib scikit-learn torch-geometric')

# %% 1 — IMPORTS AND GOOGLE DRIVE
from __future__ import annotations

import os
import glob
import gc
import json
import time
import copy
import random
import warnings
import importlib.metadata as metadata
import psutil

try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import butter, sosfiltfilt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

import mne

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import add_self_loops, coalesce

from esinet import Simulation
from esinet import Net

warnings.filterwarnings("default")
plt.switch_backend("agg")
mne.set_log_level("WARNING")

# ============================================================
# 1. REPRODUCIBILITY AND CONFIGURATION
# ============================================================
SEEDS = (42, 52, 62, 72, 82)

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

set_global_seed(SEEDS[0])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def assert_finite_array(name, value):
    if isinstance(value, torch.Tensor):
        finite = torch.isfinite(value).all().item()
        shape = tuple(value.shape)
    else:
        value = np.asarray(value)
        finite = bool(np.isfinite(value).all())
        shape = value.shape
    if not finite:
        raise RuntimeError(f"{name} contains non-finite values; shape={shape}")

# ------------------------------------------------------------
# Google Drive paths
# ------------------------------------------------------------
DRIVE_ROOT = "/content/drive/MyDrive/Esinet"
MNE_ROOT = os.path.join(DRIVE_ROOT, "mne_data")
OUTPUT_DIR = os.path.join(DRIVE_ROOT, "real_mne_sample_final")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIKHONOV_CNN_PATH = os.path.join(OUTPUT_DIR, "best_tikhonov_cnn_model.pt")
GRAPH_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_sparse_st_graph_model.pt")
PHYSICS_GAT_PATH = os.path.join(OUTPUT_DIR, "best_full_physics_gat_model.pt")
CONVDIP_MODEL_PATH = os.path.join(OUTPUT_DIR, "convdip_real_subject_model")

TRAINING_HISTORY_PATH = os.path.join(OUTPUT_DIR, "sparse_graph_training_history.csv")
REAL_RESULTS_PATH = os.path.join(OUTPUT_DIR, "real_data_results.csv")
REAL_TIMING_PATH = os.path.join(OUTPUT_DIR, "real_data_inference_times.csv")
CONFIG_PATH = os.path.join(OUTPUT_DIR, "configuration.json")
SYNTHETIC_RESULTS_PATH = os.path.join(OUTPUT_DIR, "synthetic_results.csv")

# ------------------------------------------------------------
# Source-space and simulation settings
# ------------------------------------------------------------
SMOKE_TEST = False

SOURCE_SPACING = "ico3"
TARGET_SNR = 3.0
OOD_SNR = -10.0
SOURCE_EXTENTS = (10, 20)
OOD_EXTENTS = (28, 42)

NUM_SIMULATED_SOURCES = 3
OOD_NUM_SIMULATED_SOURCES = 6

N_TRAIN = 8 if SMOKE_TEST else 40000
N_VALIDATION = 4 if SMOKE_TEST else 1000
N_TEST = 4 if SMOKE_TEST else 1000

SIMULATION_DURATION = 0.30
RESAMPLE_FREQUENCY = 100.0
N_TIMES = int(round(SIMULATION_DURATION * RESAMPLE_FREQUENCY)) + 1

MAX_EPOCHS = 1 if SMOKE_TEST else 150
BATCH_SIZE = 2 if SMOKE_TEST else 64
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 20
SCHEDULER_PATIENCE = 7

ACTIVE_WEIGHT = 50.0
ACTIVE_THRESHOLD_RATIO = 0.10
SPATIAL_LOSS_WEIGHT = 1e-4

TIKHONOV_RELATIVE_REGULARIZATION = 1e-2

EPOCH_TMIN = -0.20
EPOCH_TMAX = 0.50
INFERENCE_TMIN = 0.00
INFERENCE_TMAX = 0.30

NONLOCAL_K = 4
FUNCTIONAL_K = 3
HIDDEN_DIM = 32
HEADS = 4
DROPOUT = 0.15

if HIDDEN_DIM % HEADS != 0:
    raise ValueError("The hidden dimension must be divisible by the number of attention heads.")

# ============================================================
# 2. DISCOVER REAL MNE SAMPLE FILES IN GOOGLE DRIVE
# ============================================================
def find_all(patterns):
    results = []
    for pattern in patterns:
        results.extend(glob.glob(pattern, recursive=True))
    return sorted(set(results))

def find_first(patterns):
    results = find_all(patterns)
    if len(results) == 0: return None
    return results[0]

def find_subjects_directory(root_directory):
    possible_directories = find_all([os.path.join(root_directory, "**", "subjects")])
    possible_directories.append(root_directory)
    for directory in possible_directories:
        sample_surf = os.path.join(directory, "sample", "surf")
        sample_mri = os.path.join(directory, "sample", "mri")
        if os.path.isdir(sample_surf) and os.path.isdir(sample_mri): return directory
    return None

raw_file = find_first([os.path.join(MNE_ROOT, "**", "sample_audvis_filt-0-40_raw.fif"), os.path.join(MNE_ROOT, "**", "sample_audvis_raw.fif")])
event_file = find_first([os.path.join(MNE_ROOT, "**", "sample_audvis_raw-eve.fif"), os.path.join(MNE_ROOT, "**", "*audvis*-eve.fif")])
trans_file = find_first([os.path.join(MNE_ROOT, "**", "sample_audvis_raw-trans.fif"), os.path.join(MNE_ROOT, "**", "*audvis*-trans.fif"), os.path.join(MNE_ROOT, "**", "*-trans.fif")])
bem_file = find_first([os.path.join(MNE_ROOT, "**", "sample-5120-5120-5120-bem-sol.fif"), os.path.join(MNE_ROOT, "**", "sample-*-bem-sol.fif"), os.path.join(MNE_ROOT, "**", "*-bem-sol.fif")])
subjects_dir = find_subjects_directory(MNE_ROOT)

required_files = {"Raw file": raw_file, "Transformation file": trans_file, "BEM file": bem_file, "Subjects directory": subjects_dir}
for description, path in required_files.items():
    if path is None: raise FileNotFoundError(f"{description} was not found under {MNE_ROOT}.")

os.environ["SUBJECTS_DIR"] = subjects_dir
mne.set_config("SUBJECTS_DIR", subjects_dir, set_env=True)

# ============================================================
# 3. LOAD AND PREPROCESS REAL EEG
# ============================================================
raw_full = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)
raw = raw_full.copy().pick(picks=["eeg", "eog", "stim"], exclude="bads")
raw.apply_proj()
raw.filter(l_freq=1.0, h_freq=40.0, picks="eeg", method="fir", phase="zero", verbose=False)
raw.set_eeg_reference(ref_channels="average", projection=True, verbose=False)
raw.apply_proj()

# ============================================================
# 4. EXTRACT REAL EVENTS
# ============================================================
try:
    events = mne.find_events(raw_full, stim_channel="STI 014", shortest_event=1, verbose=False)
except Exception:
    events = mne.read_events(event_file)
    events = events[(events[:, 0] >= raw.first_samp) & (events[:, 0] <= raw.last_samp)]

event_id = {"Auditory_Left": 1, "Auditory_Right": 2, "Visual_Left": 3, "Visual_Right": 4}
present_events = {}
for event_name, event_code in event_id.items():
    count = int(np.sum(events[:, 2] == event_code))
    if count > 0: present_events[event_name] = event_code

if not present_events: raise ValueError("CRITICAL ERROR: No events found!")
event_id = present_events

# ============================================================
# 5. CREATE REAL EEG EPOCHS
# ============================================================
eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude="bads")
epochs = mne.Epochs(
    raw=raw, events=events, event_id=event_id, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
    baseline=(EPOCH_TMIN, 0.0), picks=eeg_picks, reject={"eeg": 150e-6}, preload=True, proj=True, detrend=1,
    reject_by_annotation=True, on_missing="warn", verbose=False
)
epochs.resample(RESAMPLE_FREQUENCY, npad="auto", verbose=False)

# ============================================================
# 6. CREATE REAL EVOKED RESPONSES
# ============================================================
evoked_conditions = {}
for condition_name in event_id:
    evoked_conditions[condition_name] = epochs[condition_name].average().crop(tmin=INFERENCE_TMIN, tmax=INFERENCE_TMAX)

if {"Auditory_Left", "Auditory_Right"}.issubset(evoked_conditions):
    evoked_conditions["Auditory_Average"] = mne.combine_evoked([evoked_conditions["Auditory_Left"], evoked_conditions["Auditory_Right"]], weights="equal")

if {"Visual_Left", "Visual_Right"}.issubset(evoked_conditions):
    evoked_conditions["Visual_Average"] = mne.combine_evoked([evoked_conditions["Visual_Left"], evoked_conditions["Visual_Right"]], weights="equal")

del raw_full
gc.collect()

# ============================================================
# 7. BUILD A SUBJECT-SPECIFIC REAL FORWARD MODEL
# ============================================================
source_space = mne.setup_source_space(subject="sample", spacing=SOURCE_SPACING, subjects_dir=subjects_dir, add_dist=False, verbose=False)
fwd_free = mne.make_forward_solution(info=epochs.info, trans=trans_file, src=source_space, bem=bem_file, meg=False, eeg=True, mindist=5.0, n_jobs=1, verbose=False)
fwd_free = mne.pick_types_forward(fwd_free, meg=False, eeg=True, ref_meg=False, exclude="bads")

common_channels = [ch for ch in epochs.ch_names if ch in fwd_free["info"]["ch_names"]]
epochs.pick(common_channels)
for c in evoked_conditions: evoked_conditions[c].pick(common_channels)

fwd_free = mne.pick_channels_forward(fwd_free, include=common_channels, ordered=True, copy=True)
forward_channel_order = fwd_free["info"]["ch_names"]
epochs.reorder_channels(forward_channel_order)
for c in evoked_conditions: evoked_conditions[c].reorder_channels(forward_channel_order)

fwd_fixed = mne.convert_forward_solution(fwd_free, surf_ori=True, force_fixed=True, use_cps=True, copy=True, verbose=False)
lead_field = np.asarray(fwd_fixed["sol"]["data"], dtype=np.float64)
num_vertices = sum(len(src["vertno"]) for src in fwd_fixed["src"])
num_channels = len(fwd_fixed["info"]["ch_names"])
source_coordinates_mm = np.vstack([src["rr"][src["vertno"]] for src in fwd_fixed["src"]]).astype(np.float64) * 1000.0

# ============================================================
# 8. CORTICAL MESH GRAPH & PHYSICS-GAT GRAPH COMPONENTS
# ============================================================
def build_cortical_edges(forward_model):
    edge_set = set()
    offset = 0
    for hemisphere in forward_model["src"]:
        used_vertices = np.asarray(hemisphere["vertno"], dtype=np.int64)
        vertex_to_local = {int(v): i for i, v in enumerate(used_vertices)}
        triangles = hemisphere.get("use_tris", hemisphere.get("tris", None))
        for triangle in triangles:
            triangle = [int(v) for v in triangle]
            if not all(v in vertex_to_local for v in triangle): continue
            local_nodes = [vertex_to_local[v] + offset for v in triangle]
            pairs = [(local_nodes[0], local_nodes[1]), (local_nodes[1], local_nodes[2]), (local_nodes[2], local_nodes[0])]
            for first_node, second_node in pairs:
                if first_node != second_node:
                    edge_set.add(tuple(sorted((first_node, second_node))))
        offset += len(used_vertices)
    if len(edge_set) == 0: raise RuntimeError("The cortical graph contains no edges.")
    return np.array(sorted(edge_set), dtype=np.int64).T

edge_index_np = build_cortical_edges(fwd_fixed)

def make_normalized_sparse_adjacency(edge_index, number_of_nodes, device):
    source, target = edge_index[0], edge_index[1]
    self_nodes = np.arange(number_of_nodes, dtype=np.int64)
    rows = np.concatenate([source, target, self_nodes])
    columns = np.concatenate([target, source, self_nodes])
    degree = np.maximum(np.bincount(rows, minlength=number_of_nodes).astype(np.float32), 1.0)
    values = (1.0 / np.sqrt(degree[rows] * degree[columns])).astype(np.float32)
    indices = torch.tensor(np.vstack([rows, columns]), dtype=torch.long, device=device)
    values_tensor = torch.tensor(values, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices=indices, values=values_tensor, size=(number_of_nodes, number_of_nodes), device=device).coalesce()

sparse_adjacency = make_normalized_sparse_adjacency(edge_index_np, num_vertices, device)
edge_index_tensor = torch.tensor(edge_index_np, dtype=torch.long, device=device)

# --- PHYSICS-GAT GRAPH COMPONENTS ---
def compute_geodesic_matrix(undirected_edges: np.ndarray, coordinates_mm: np.ndarray) -> np.ndarray:
    first, second = undirected_edges
    edge_lengths = np.linalg.norm(coordinates_mm[first] - coordinates_mm[second], axis=1)
    sparse_graph = coo_matrix(
        (np.concatenate([edge_lengths, edge_lengths]), (np.concatenate([first, second]), np.concatenate([second, first]))),
        shape=(len(coordinates_mm), len(coordinates_mm))
    ).tocsr()
    distances = np.asarray(dijkstra(sparse_graph, directed=False), dtype=np.float32)
    finite_values = distances[np.isfinite(distances)]
    if finite_values.size == 0: raise RuntimeError("No finite geodesic distance was computed.")
    distances[~np.isfinite(distances)] = finite_values.max() + 100.0
    return distances

GEODESIC_MM = compute_geodesic_matrix(edge_index_np, source_coordinates_mm)

def make_local_edge_index() -> torch.Tensor:
    first, second = edge_index_np
    edge_index = torch.tensor(np.vstack([np.concatenate([first, second]), np.concatenate([second, first])]), dtype=torch.long)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_vertices)
    return coalesce(edge_index, num_nodes=num_vertices)

LOCAL_EDGE_INDEX = make_local_edge_index()

def normalized_leadfield_profiles(lead_field: np.ndarray) -> np.ndarray:
    profiles = np.asarray(lead_field.T, dtype=np.float32).copy()
    profiles -= profiles.mean(axis=1, keepdims=True)
    profiles /= np.linalg.norm(profiles, axis=1, keepdims=True) + 1e-8
    return profiles

if not 1 <= NONLOCAL_K < num_vertices: raise ValueError("NONLOCAL_K must be between 1 and V-1.")
if not 1 <= FUNCTIONAL_K < num_vertices: raise ValueError("FUNCTIONAL_K must be between 1 and V-1.")

def create_base_graph(lead_field: np.ndarray):
    profiles = normalized_leadfield_profiles(lead_field)
    similarity = np.abs(profiles @ profiles.T)
    np.fill_diagonal(similarity, -np.inf)
    neighbours = np.argpartition(similarity, -NONLOCAL_K, axis=1)[:, -NONLOCAL_K:]
    source = np.repeat(np.arange(num_vertices), NONLOCAL_K)
    target = neighbours.reshape(-1)
    nonlocal_edges = torch.tensor(np.vstack([source, target]), dtype=torch.long)
    edge_index = torch.cat([LOCAL_EDGE_INDEX, nonlocal_edges, nonlocal_edges.flip(0)], dim=1)
    return coalesce(edge_index, num_nodes=num_vertices), profiles

def create_edge_attributes(edge_index: torch.Tensor, leadfield_profiles: np.ndarray, functional_similarity=None) -> torch.Tensor:
    first = edge_index[0].cpu().numpy()
    second = edge_index[1].cpu().numpy()
    euclidean = np.linalg.norm(source_coordinates_mm[first] - source_coordinates_mm[second], axis=1) / 200.0
    geodesic = GEODESIC_MM[first, second] / 300.0
    leadfield_similarity = np.abs(np.sum(leadfield_profiles[first] * leadfield_profiles[second], axis=1))

    if functional_similarity is None:
        functional = np.zeros(len(first), dtype=np.float32)
    else:
        functional = functional_similarity[first, second].astype(np.float32)

    self_mask = first == second
    euclidean[self_mask] = 0.0
    geodesic[self_mask] = 0.0
    leadfield_similarity[self_mask] = 1.0
    functional[self_mask] = 0.0

    values = np.column_stack([euclidean, geodesic, leadfield_similarity, functional]).astype(np.float32)
    if not np.isfinite(values).all():
        invalid_count = int((~np.isfinite(values)).sum())
        raise RuntimeError(f"Non-finite edge attributes detected: {invalid_count}")
    if values.shape != (edge_index.shape[1], 4):
        raise RuntimeError("Edge-attribute shape is inconsistent with edge_index.")
    return torch.from_numpy(values)

class GraphBuilder:
    def __init__(self, lead_field: np.ndarray, dynamic: bool):
        self.dynamic = dynamic
        self.base_edges, self.profiles = create_base_graph(lead_field)
        self.static_attributes = create_edge_attributes(self.base_edges, self.profiles)

    def __call__(self, initial_source: np.ndarray):
        if not self.dynamic: return self.base_edges, self.static_attributes
        signal = initial_source - initial_source.mean(axis=1, keepdims=True)
        signal /= np.linalg.norm(signal, axis=1, keepdims=True) + 1e-8

        functional_similarity = np.abs(signal @ signal.T).astype(np.float32)
        selection_similarity = functional_similarity.copy()
        np.fill_diagonal(selection_similarity, -np.inf)

        neighbours = np.argpartition(selection_similarity, -FUNCTIONAL_K, axis=1)[:, -FUNCTIONAL_K:]
        source = np.repeat(np.arange(num_vertices), FUNCTIONAL_K)
        target = neighbours.reshape(-1)
        functional_edges = torch.tensor(np.vstack([source, target]), dtype=torch.long)
        final_edges = torch.cat([self.base_edges, functional_edges, functional_edges.flip(0)], dim=1)
        final_edges = coalesce(final_edges, num_nodes=num_vertices)

        np.fill_diagonal(functional_similarity, 0.0)
        final_attributes = create_edge_attributes(final_edges, self.profiles, functional_similarity)
        return final_edges, final_attributes

STATIC_GRAPH_BUILDER = GraphBuilder(lead_field, dynamic=False)
DYNAMIC_GRAPH_BUILDER = GraphBuilder(lead_field, dynamic=True)

# ============================================================
# 9. EXTRACTION UTILITIES
# ============================================================
def extract_array(item):
    if hasattr(item, "get_data"): return np.asarray(item.get_data())
    if hasattr(item, "data"): return np.asarray(item.data)
    return np.asarray(item)

def ensure_two_dimensions(array):
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[0] == 1: array = array[0]
    return array.astype(np.float32, copy=False)

def extract_eeg_collection(data):
    if isinstance(data, list) or isinstance(data, tuple):
        return np.stack([ensure_two_dimensions(extract_array(data[i])) for i in range(len(data))], axis=0)
    return np.stack([ensure_two_dimensions(extract_array(data))], axis=0)

def extract_source_full(data):
    if isinstance(data, list) or isinstance(data, tuple):
        return np.stack([ensure_two_dimensions(extract_array(data[i])) for i in range(len(data))], axis=0).astype(np.float32)
    return np.stack([ensure_two_dimensions(extract_array(data))], axis=0).astype(np.float32)

def extract_test(sim):
    return extract_eeg_collection(sim.eeg_data), extract_source_full(sim.source_data)

# ============================================================
# 10. NEURAL ARCHITECTURES
# ============================================================

# 10A. TikhonovTemporalCNN
class TikhonovTemporalCNN(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.GroupNorm(2, 8),
            nn.GELU(),
            nn.Conv1d(8, 12, kernel_size=3, padding=1),
            nn.GroupNorm(3, 12),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten()
        )
        self.temporal_decoder = nn.Sequential(
            nn.Linear(48, hidden),
            nn.GELU(),
            nn.Linear(hidden, N_TIMES)
        )
    def forward(self, source_time_series, *args):
        batch_size, vertices, times = source_time_series.shape
        temporal_input = source_time_series.reshape(batch_size * vertices, 1, times)
        features = self.temporal_encoder(temporal_input)
        output = self.temporal_decoder(features)
        return output.reshape(batch_size, vertices, -1)

# 10B. Sparse Spatio-Temporal Graph Network
class SparseSpatioTemporalGraphNet(nn.Module):
    def __init__(self, hidden_dimension=32, dropout=0.20):
        super().__init__()
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.1),
            nn.Conv1d(8, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(4)
        )
        temporal_features = 8 * 4
        self.input_projection = nn.Linear(temporal_features, hidden_dimension)
        self.graph_projection_1 = nn.Linear(hidden_dimension, hidden_dimension)
        self.graph_projection_2 = nn.Linear(hidden_dimension, hidden_dimension)
        self.node_gate = nn.Sequential(nn.Linear(hidden_dimension, hidden_dimension), nn.Sigmoid())
        self.norm_1 = nn.LayerNorm(hidden_dimension)
        self.norm_2 = nn.LayerNorm(hidden_dimension)
        self.temporal_decoder = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, N_TIMES)
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def sparse_graph_propagation(node_features, adjacency):
        batch_size, vertices, features = node_features.shape
        matrix = node_features.permute(1, 0, 2).reshape(vertices, batch_size * features)
        propagated = torch.sparse.mm(adjacency, matrix)
        return propagated.reshape(vertices, batch_size, features).permute(1, 0, 2)

    def forward(self, source_time_series, adjacency, *args):
        batch_size, vertices, times = source_time_series.shape
        temporal_input = source_time_series.reshape(batch_size * vertices, 1, times)
        temporal_features = self.temporal_encoder(temporal_input).reshape(batch_size, vertices, -1)
        hidden = self.input_projection(temporal_features)

        propagated_1 = self.sparse_graph_propagation(hidden, adjacency)
        propagated_1 = self.graph_projection_1(propagated_1)
        hidden = self.norm_1(hidden + propagated_1)
        hidden = F.leaky_relu(hidden, negative_slope=0.1)
        hidden = self.dropout(hidden)

        gate = self.node_gate(hidden)
        hidden = hidden * gate

        propagated_2 = self.sparse_graph_propagation(hidden, adjacency)
        propagated_2 = self.graph_projection_2(propagated_2)
        hidden = self.norm_2(hidden + propagated_2)
        hidden = F.leaky_relu(hidden, negative_slope=0.1)
        hidden = self.dropout(hidden)

        output = self.temporal_decoder(hidden)
        return output

# 10C. Full Physics GAT
class TemporalEncoderPhysics(nn.Module):
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
        if signal.ndim != 2: raise ValueError("The temporal encoder expects [V, T].")
        features = self.network(signal.unsqueeze(1)).flatten(1)
        return self.projection(features)

class EdgeAwareGATBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.gat = GATv2Conv(hidden, hidden // HEADS, heads=HEADS, edge_dim=4, dropout=DROPOUT, add_self_loops=False)
        self.normalization = nn.LayerNorm(hidden)

    def forward(self, features, edge_index, edge_attr):
        message = self.gat(features, edge_index, edge_attr)
        message = F.dropout(F.gelu(message), p=DROPOUT, training=self.training)
        return self.normalization(features + message)

class FullPhysicsGAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_encoder = TemporalEncoderPhysics(HIDDEN_DIM)
        self.graph_block_1 = EdgeAwareGATBlock(HIDDEN_DIM)
        self.graph_block_2 = EdgeAwareGATBlock(HIDDEN_DIM)
        self.temporal_decoder = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, N_TIMES)
        )

    def forward(self, data, *args):
        initial_source, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        features = self.temporal_encoder(initial_source)
        features = self.graph_block_1(features, edge_index, edge_attr)
        features = self.graph_block_2(features, edge_index, edge_attr)
        output = self.temporal_decoder(features)
        batch_size = batch.max().item() + 1
        return output.reshape(batch_size, num_vertices, -1)

# ============================================================
# 11. LOSS TENSORS
# ============================================================
GEODESIC_TENSOR = torch.tensor(GEODESIC_MM, dtype=torch.float32, device=device)
LEAD_FIELD_TENSOR = torch.tensor(lead_field, dtype=torch.float32, device=device)

def graph_training_loss(prediction, target, eeg_target, graph_edges, active_weight, active_threshold_ratio, spatial_weight, forward_weight=1.0):
    if prediction.shape != target.shape: raise ValueError(f"Prediction shape {prediction.shape} does not match target shape {target.shape}.")

    waveform_loss = F.mse_loss(prediction, target)

    predicted_map = prediction.abs().amax(dim=-1)
    target_map = target.abs().amax(dim=-1)

    target_peak = target_map.amax(dim=1, keepdim=True).clamp_min(1e-8)
    active_target = (target_map >= active_threshold_ratio * target_peak)
    weights = 1.0 + (active_weight - 1.0) * active_target.float()

    map_loss = (weights * (predicted_map - target_map).pow(2)).sum() / weights.sum().clamp_min(1.0)

    residual_map = predicted_map - target_map
    edge_smoothness = (residual_map[:, graph_edges[0]] - residual_map[:, graph_edges[1]]).square().mean()

    probability = predicted_map / predicted_map.sum(dim=1, keepdim=True).clamp_min(1e-8)
    target_probability = target_map / target_map.sum(dim=1, keepdim=True).clamp_min(1e-8)

    geodesic_terms_p2t = []
    geodesic_terms_t2p = []
    for i in range(len(prediction)):
        true_support = torch.where(active_target[i])[0]
        if len(true_support) == 0: true_support = target_map[i].argmax().reshape(1)

        # Differentiable softmin for target-to-prediction support
        d_p2t = GEODESIC_TENSOR[:, true_support].amin(dim=1)
        tau = 10.0
        softmin_d = -tau * torch.logsumexp(-GEODESIC_TENSOR / tau + torch.log(probability[i] + 1e-12).unsqueeze(0), dim=1)

        geodesic_terms_p2t.append((probability[i] * d_p2t).sum() / 100.0)
        geodesic_terms_t2p.append((target_probability[i] * softmin_d).sum() / 100.0)

    geodesic_loss = (torch.stack(geodesic_terms_p2t).mean() + torch.stack(geodesic_terms_t2p).mean()) / 2.0

    reconstructed_eeg = torch.einsum("cv,bvt->bct", LEAD_FIELD_TENSOR, prediction)
    prediction_physical = prediction * Y_TARGET_SCALE
    reconstructed_eeg = torch.einsum("cv,bvt->bct", LEAD_FIELD_TENSOR, prediction_physical)
    eeg_physical = eeg_target * X_INPUT_SCALE

    forward_loss = ((reconstructed_eeg - eeg_physical).square().mean() / eeg_physical.square().mean().clamp_min(1e-12))

    total = waveform_loss + map_loss + spatial_weight * edge_smoothness + 2e-3 * geodesic_loss + forward_weight * forward_loss
    return total, waveform_loss.detach(), map_loss.detach()

# ============================================================
# 12. DATA LOADERS (USING PYTORCH GEOMETRIC BATCHING)
# ============================================================
class PhysicsDataset(Dataset):
    def __init__(self, x_data, y_data, build_dynamic_graph=False):
        self.x = x_data
        self.y = y_data
        self.build_dynamic_graph = build_dynamic_graph

    def __len__(self): return len(self.x)

    def __getitem__(self, idx):
        source_ts = self.x[idx]
        target = self.y[idx]

        if self.build_dynamic_graph:
            edge_index, edge_attr = DYNAMIC_GRAPH_BUILDER(source_ts)
        else:
            edge_index, edge_attr = None, None

        return Data(
            x=torch.from_numpy(source_ts),
            y=torch.from_numpy(target),
            edge_index=edge_index,
            edge_attr=edge_attr
        )

def execute_epoch(model_name, model, loader, optimizer, training):
    if training: model.train()
    else: model.eval()

    total_loss, total_wave, total_map, total_samples = 0.0, 0.0, 0.0, 0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch in loader:
            batch_size = batch.num_graphs

            if model_name == "full_physics_gat":
                batch = batch.to(device)
                batch_x = batch.x.reshape(batch_size, num_vertices, N_TIMES)
                batch_y = batch.y.reshape(batch_size, num_vertices, N_TIMES)
                prediction = model(batch)
            else:
                batch_x = batch.x.reshape(batch_size, num_vertices, N_TIMES).to(device, non_blocking=True)
                batch_y = batch.y.reshape(batch_size, num_vertices, N_TIMES).to(device, non_blocking=True)
                if model_name == "sparse_st_graph": prediction = model(batch_x, sparse_adjacency)
                elif model_name == "tikhonov_cnn": prediction = model(batch_x)

            if training: optimizer.zero_grad(set_to_none=True)

            loss, wave_l, map_l = graph_training_loss(prediction, batch_y, batch_x, edge_index_tensor, ACTIVE_WEIGHT, ACTIVE_THRESHOLD_RATIO, SPATIAL_LOSS_WEIGHT)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            current_batch_size = batch_x.shape[0]
            total_loss += loss.item() * current_batch_size
            total_wave += wave_l.item() * current_batch_size
            total_map += map_l.item() * current_batch_size
            total_samples += current_batch_size

    return {"loss": total_loss / total_samples, "wave": total_wave / total_samples, "map": total_map / total_samples}

# ============================================================
# 13. SEED LOOP
# ============================================================
all_synthetic_results = []
all_real_results = []
all_timing_results = []

for seed_idx, current_seed in enumerate(SEEDS):
    print(f"\n{'='*75}\nSTARTING PIPELINE FOR SEED {current_seed}\n{'='*75}")
    set_global_seed(current_seed)

    def make_simulation(n_samples, seed_val, snr, extents, sources=3):
        set_global_seed(seed_val)
        settings = {"duration_of_trial": SIMULATION_DURATION, "number_of_sources": sources, "extents": extents, "target_snr": snr}
        simulation = Simulation(fwd_fixed, epochs.info.copy(), settings=settings)
        simulation.simulate(n_samples=n_samples)
        return simulation

    sim_train = make_simulation(N_TRAIN, current_seed, TARGET_SNR, SOURCE_EXTENTS, sources=NUM_SIMULATED_SOURCES)
    sim_validation = make_simulation(N_VALIDATION, current_seed + 1, TARGET_SNR, SOURCE_EXTENTS, sources=NUM_SIMULATED_SOURCES)

    sim_test_id = make_simulation(N_TEST, current_seed + 100, TARGET_SNR, SOURCE_EXTENTS, sources=NUM_SIMULATED_SOURCES)
    sim_test_ood_snr = make_simulation(N_TEST, current_seed + 101, OOD_SNR, SOURCE_EXTENTS, sources=NUM_SIMULATED_SOURCES)
    sim_test_ood_extent = make_simulation(N_TEST, current_seed + 102, TARGET_SNR, OOD_EXTENTS, sources=NUM_SIMULATED_SOURCES)
    sim_test_ood_sources = make_simulation(N_TEST, current_seed + 103, TARGET_SNR, SOURCE_EXTENTS, sources=OOD_NUM_SIMULATED_SOURCES)

    X_eeg_train = extract_eeg_collection(sim_train.eeg_data)
    Y_source_train = extract_source_full(sim_train.source_data)
    if X_eeg_train.shape[-1] != N_TIMES: raise RuntimeError(f"Unexpected EEG time dimension: expected {N_TIMES}, received {X_eeg_train.shape[-1]}")
    if Y_source_train.shape[-1] != N_TIMES: raise RuntimeError(f"Unexpected Source time dimension: expected {N_TIMES}, received {Y_source_train.shape[-1]}")

    X_eeg_validation = extract_eeg_collection(sim_validation.eeg_data)
    Y_source_validation = extract_source_full(sim_validation.source_data)

    X_eeg_test_id, Y_source_test_id = extract_test(sim_test_id)
    X_eeg_test_ood_snr, Y_source_test_ood_snr = extract_test(sim_test_ood_snr)
    X_eeg_test_ood_extent, Y_source_test_ood_extent = extract_test(sim_test_ood_extent)
    X_eeg_test_ood_sources, Y_source_test_ood_sources = extract_test(sim_test_ood_sources)

    def compute_tikhonov_inverse(forward_matrix, relative_regularization):
        K = np.asarray(forward_matrix, dtype=np.float64)
        sensor_gram = K @ K.T
        average_eigenvalue = np.trace(sensor_gram) / sensor_gram.shape[0]
        regularization = relative_regularization * average_eigenvalue
        regularized_gram = sensor_gram + regularization * np.eye(sensor_gram.shape[0])
        inverse_operator = K.T @ np.linalg.solve(regularized_gram, np.eye(regularized_gram.shape[0]))
        return inverse_operator.astype(np.float32), float(regularization)

    K_dagger, effective_regularization = compute_tikhonov_inverse(lead_field, TIKHONOV_RELATIVE_REGULARIZATION)

    def apply_tikhonov_batch(eeg_batch, inverse_operator):
        return np.einsum("vc,bct->bvt", inverse_operator, eeg_batch, optimize=True).astype(np.float32)

    X_source_train = apply_tikhonov_batch(X_eeg_train, K_dagger)
    X_source_validation = apply_tikhonov_batch(X_eeg_validation, K_dagger)
    X_source_test_id = apply_tikhonov_batch(X_eeg_test_id, K_dagger)
    X_source_test_ood_snr = apply_tikhonov_batch(X_eeg_test_ood_snr, K_dagger)
    X_source_test_ood_extent = apply_tikhonov_batch(X_eeg_test_ood_extent, K_dagger)
    X_source_test_ood_sources = apply_tikhonov_batch(X_eeg_test_ood_sources, K_dagger)

    def robust_scale(array, percentile=99.5):
        value = np.percentile(np.abs(array), percentile)
        return float(max(value, 1e-12))

    X_INPUT_SCALE = robust_scale(X_source_train)
    Y_TARGET_SCALE = robust_scale(Y_source_train)

    def normalize_input(array): return np.clip(array / X_INPUT_SCALE, -10.0, 10.0).astype(np.float32)
    def normalize_target(array): return np.clip(array / Y_TARGET_SCALE, -10.0, 10.0).astype(np.float32)

    X_source_train = normalize_input(X_source_train)
    X_source_validation = normalize_input(X_source_validation)
    X_source_test_id = normalize_input(X_source_test_id)
    X_source_test_ood_snr = normalize_input(X_source_test_ood_snr)
    X_source_test_ood_extent = normalize_input(X_source_test_ood_extent)
    X_source_test_ood_sources = normalize_input(X_source_test_ood_sources)

    Y_source_train = normalize_target(Y_source_train)
    Y_source_validation = normalize_target(Y_source_validation)
    Y_source_test_id = normalize_target(Y_source_test_id)
    Y_source_test_ood_snr = normalize_target(Y_source_test_ood_snr)
    Y_source_test_ood_extent = normalize_target(Y_source_test_ood_extent)
    Y_source_test_ood_sources = normalize_target(Y_source_test_ood_sources)

    train_loader_graph = PyGDataLoader(PhysicsDataset(X_source_train, Y_source_train, build_dynamic_graph=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader_graph = PyGDataLoader(PhysicsDataset(X_source_validation, Y_source_validation, build_dynamic_graph=True), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    train_loader_cnn = PyGDataLoader(PhysicsDataset(X_source_train, Y_source_train, build_dynamic_graph=False), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader_cnn = PyGDataLoader(PhysicsDataset(X_source_validation, Y_source_validation, build_dynamic_graph=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Clean memory of large arrays before training begins
    del X_eeg_train, Y_source_train, X_eeg_validation, Y_source_validation
    gc.collect()

    tikhonov_cnn_model = TikhonovTemporalCNN().to(device)
    sparse_st_model = SparseSpatioTemporalGraphNet(hidden_dimension=32, dropout=0.20).to(device)
    physics_gat_model = FullPhysicsGAT().to(device)

    # SMOKE TEST
    if seed_idx == 0:
        print("\n" + "=" * 75)
        print("EXECUTING SMOKE TEST")
        print("=" * 75)
        test_batch = next(iter(train_loader_graph))
        test_x = test_batch.x.reshape(test_batch.num_graphs, num_vertices, N_TIMES).to(device)
        test_y = test_batch.y.reshape(test_batch.num_graphs, num_vertices, N_TIMES).to(device)
        test_batch = test_batch.to(device)

        assert_finite_array("Smoke test batch_x", test_x)
        assert_finite_array("Smoke test batch_y", test_y)

        with torch.no_grad():
            pred_physics = physics_gat_model(test_batch)
            assert_finite_array("Smoke test PhysicsGAT prediction", pred_physics)
            pred_sparse = sparse_st_model(test_x, sparse_adjacency)
            assert_finite_array("Smoke test SparseST prediction", pred_sparse)
            pred_cnn = tikhonov_cnn_model(test_x)
            assert_finite_array("Smoke test CNN prediction", pred_cnn)

        physics_gat_model.train()
        pred_physics = physics_gat_model(test_batch)
        test_loss, _, _ = graph_training_loss(pred_physics, test_y, test_x, edge_index_tensor, ACTIVE_WEIGHT, ACTIVE_THRESHOLD_RATIO, SPATIAL_LOSS_WEIGHT)
        assert_finite_array("Smoke test loss", test_loss)
        test_loss.backward()
        print("Forward and backward smoke test passed.\n")

    def train_model_loop(model_name, model, loader_train, loader_val, save_path):
        print(f"\n{'='*75}\nTRAINING: {model_name.upper()}\n{'='*75}")
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE, min_lr=1e-6)

        history = []
        best_validation_loss = float("inf")
        epochs_without_improvement = 0
        start_epoch = 1
        start_time = time.time()

        last_save_path = save_path.replace("best", "last")
        if os.path.exists(last_save_path):
            chkpt = torch.load(last_save_path, map_location=device, weights_only=False)
            model.load_state_dict(chkpt["model_state_dict"])
            optimizer.load_state_dict(chkpt["optimizer_state_dict"])
            scheduler.load_state_dict(chkpt["scheduler_state_dict"])
            start_epoch = chkpt["epoch"] + 1
            best_validation_loss = chkpt["best_validation_loss"]
            epochs_without_improvement = chkpt["epochs_without_improvement"]
            history = chkpt.get("history", [])
            random.setstate(chkpt["python_rng"])
            np.random.set_state(chkpt["numpy_rng"])
            torch.set_rng_state(chkpt["torch_rng"])
            if torch.cuda.is_available() and chkpt["cuda_rng"] is not None: torch.cuda.set_rng_state_all(chkpt["cuda_rng"])
            print(f"Resumed from epoch {start_epoch - 1}")

        for epoch in range(start_epoch, MAX_EPOCHS + 1):
            train_stats = execute_epoch(model_name, model, loader_train, optimizer, training=True)
            val_stats = execute_epoch(model_name, model, loader_val, optimizer, training=False)

            validation_loss = val_stats["loss"]
            scheduler.step(validation_loss)
            learning_rate = optimizer.param_groups[0]["lr"]

            history.append({
                "Epoch": epoch, "Training loss": train_stats["loss"], "Validation loss": val_stats["loss"],
                "Training wave": train_stats["wave"], "Validation wave": val_stats["wave"],
                "Training map": train_stats["map"], "Validation map": val_stats["map"], "Learning rate": learning_rate
            })

            improved = validation_loss < best_validation_loss - 1e-7
            if improved:
                best_validation_loss = validation_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            checkpoint_data = {
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch,
                "best_validation_loss": best_validation_loss, "epochs_without_improvement": epochs_without_improvement,
                "input_scale": X_INPUT_SCALE, "target_scale": Y_TARGET_SCALE,
                "channel_names": epochs.ch_names, "vertices": [fwd_fixed["src"][0]["vertno"], fwd_fixed["src"][1]["vertno"]],
                "history": history, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
            }

            torch.save(checkpoint_data, last_save_path)
            if improved: torch.save(checkpoint_data, save_path)

            if epoch == 1 or epoch % 10 == 0 or improved:
                print(f"Epoch {epoch:03d} | Train={train_stats['loss']:.6f} | Val={validation_loss:.6f} | LR={learning_rate:.2e}")

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print("Early stopping at epoch", epoch)
                break

        training_minutes = (time.time() - start_time) / 60.0
        print(f"{model_name} training time: {training_minutes:.2f} minutes")

        if os.path.exists(save_path):
            checkpoint = torch.load(save_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return history, training_minutes

    cnn_path_seed = TIKHONOV_CNN_PATH.replace(".pt", f"_{current_seed}.pt")
    graph_path_seed = GRAPH_MODEL_PATH.replace(".pt", f"_{current_seed}.pt")
    physics_path_seed = PHYSICS_GAT_PATH.replace(".pt", f"_{current_seed}.pt")
    convdip_path_seed = f"{CONVDIP_MODEL_PATH}_{current_seed}"

    cnn_history, cnn_train_mins = train_model_loop("tikhonov_cnn", tikhonov_cnn_model, train_loader_cnn, val_loader_cnn, cnn_path_seed)
    sparse_history, sparse_train_mins = train_model_loop("sparse_st_graph", sparse_st_model, train_loader_cnn, val_loader_cnn, graph_path_seed)
    physics_history, physics_train_mins = train_model_loop("full_physics_gat", physics_gat_model, train_loader_graph, val_loader_graph, physics_path_seed)

    print("\n" + "=" * 75)
    print("TRAINING: CONVDIP")
    print("=" * 75)
    set_global_seed(current_seed)
    convdip_model = Net(fwd_fixed)
    convdip_training_start = time.time()
    convdip_model.fit(sim_train, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE)
    convdip_training_minutes = (time.time() - convdip_training_start) / 60.0
    try: convdip_model.save(convdip_path_seed)
    except Exception: pass

    # ============================================================
    # SYNTHETIC EVALUATION
    # ============================================================
    def maximum_absolute_map(stc_data): return np.max(np.abs(stc_data), axis=-1)
    def normalize_map(source_map):
        source_map = np.asarray(source_map, dtype=np.float64)
        return source_map / (np.max(np.abs(source_map)) + 1e-12)
    def normalized_map_mse(first_map, second_map): return float(np.mean((normalize_map(first_map) - normalize_map(second_map)) ** 2))
    def cosine_similarity(first_map, second_map): return float(1.0 - cosine(normalize_map(first_map), normalize_map(second_map)))
    def peak_index(stc_map): return int(np.argmax(stc_map))

    def evaluate_synthetic(model_name, model, x_data, y_true, condition_name):
        model.eval()
        with torch.no_grad():
            if model_name == "convdip":
                temp_sim = Simulation(fwd_fixed, epochs.info.copy())
                temp_sim.eeg_data = [mne.EvokedArray(x_data[i], epochs.info, tmin=0.0) for i in range(len(x_data))]
                try:
                    preds_list = convdip_model.predict(temp_sim)
                    pred = np.stack([p.data for p in preds_list], axis=0)
                except Exception as e:
                    raise RuntimeError("ConvDip prediction failed during synthetic evaluation.") from e
            else:
                x_tensor = torch.from_numpy(x_data).to(device)
                if model_name == "tikhonov_cnn": pred = model(x_tensor)
                elif model_name == "sparse_st_graph": pred = model(x_tensor, sparse_adjacency)
                elif model_name == "full_physics_gat":
                    ds = PhysicsDataset(x_data, y_true, build_dynamic_graph=True)
                    dl = PyGDataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
                    preds = []
                    for b in dl:
                        b = b.to(device)
                        preds.append(model(b))
                    pred = torch.cat(preds, dim=0)
                pred = pred.cpu().numpy()

        if model_name == "convdip":
            # ConvDip predicts in physical scale, so compare against unscaled physical y_true
            mse = np.mean((pred - (y_true * Y_TARGET_SCALE))**2)
        else:
            mse = np.mean((pred - y_true)**2)
        target_map = np.max(np.abs(y_true), axis=-1)
        pred_map = np.max(np.abs(pred), axis=-1)

        spearman_r, _ = spearmanr(normalize_map(target_map.mean(axis=0)), normalize_map(pred_map.mean(axis=0)))
        peak_dist = GEODESIC_MM[peak_index(target_map.mean(axis=0)), peak_index(pred_map.mean(axis=0))]

        all_synthetic_results.append({
            "Seed": current_seed, "Condition": condition_name, "Algorithm": model_name,
            "Waveform MSE": float(mse),
            "Normalized-map MSE": normalized_map_mse(target_map.mean(axis=0), pred_map.mean(axis=0)),
            "Cosine Similarity": cosine_similarity(target_map.mean(axis=0), pred_map.mean(axis=0)),
            "Peak distance (mm)": float(peak_dist)
        })

    for condition, x_val, y_val in [
        ("ID", X_source_test_id, Y_source_test_id),
        ("OOD-SNR", X_source_test_ood_snr, Y_source_test_ood_snr),
        ("OOD-Extent", X_source_test_ood_extent, Y_source_test_ood_extent),
        ("OOD-Sources", X_source_test_ood_sources, Y_source_test_ood_sources)
    ]:
        evaluate_synthetic("tikhonov_cnn", tikhonov_cnn_model, x_val, y_val, condition)
        evaluate_synthetic("sparse_st_graph", sparse_st_model, x_val, y_val, condition)
        evaluate_synthetic("full_physics_gat", physics_gat_model, x_val, y_val, condition)
        evaluate_synthetic("convdip", convdip_model, X_eeg_test_id if condition=="ID" else (X_eeg_test_ood_snr if condition=="OOD-SNR" else (X_eeg_test_ood_extent if condition=="OOD-Extent" else X_eeg_test_ood_sources)), y_val, condition)

    del sim_test_id, sim_test_ood_snr, sim_test_ood_extent, sim_test_ood_sources, sim_train
    gc.collect()

    # ============================================================
    # REAL EEG INFERENCE
    # ============================================================
    noise_covariance = mne.compute_covariance(epochs, tmin=EPOCH_TMIN, tmax=0.0, method=["shrunk", "empirical"], rank=None, verbose=False)
    dspm_inverse_operator = mne.minimum_norm.make_inverse_operator(info=epochs.info, forward=fwd_free, noise_cov=noise_covariance, loose=0.2, depth=0.8, fixed=False, rank=None, verbose=False)
    lambda2 = 1.0 / TARGET_SNR ** 2

    dspm_results = {}
    for condition_name, evoked in evoked_conditions.items():
        dspm_results[condition_name] = mne.minimum_norm.apply_inverse(evoked, dspm_inverse_operator, lambda2=lambda2, method="dSPM", pick_ori=None, verbose=False)

    def create_graph_stc(source_map): return mne.SourceEstimate(data=source_map[:, np.newaxis], vertices=[fwd_fixed["src"][0]["vertno"], fwd_fixed["src"][1]["vertno"]], tmin=0.0, tstep=1.0, subject="sample")

    @torch.no_grad()
    def predict_graph_on_real_evoked(evoked, model_name, model):
        t0 = time.perf_counter()
        eeg_data = np.asarray(evoked.data, dtype=np.float32)
        source_initialization = np.einsum("vc,ct->vt", K_dagger, eeg_data, optimize=True).astype(np.float32)
        source_initialization = np.clip(source_initialization / X_INPUT_SCALE, -10.0, 10.0)
        model_input = torch.from_numpy(source_initialization).unsqueeze(0).to(device)

        if torch.cuda.is_available(): torch.cuda.synchronize()
        if model_name == "sparse_st_graph": prediction_scaled = model(model_input, sparse_adjacency)
        elif model_name == "tikhonov_cnn": prediction_scaled = model(model_input)
        else:
            edge_index, edge_attr = DYNAMIC_GRAPH_BUILDER(source_initialization)
            ds = Data(x=model_input[0], edge_index=edge_index, edge_attr=edge_attr)
            dl = PyGDataLoader([ds], batch_size=1)
            b = next(iter(dl)).to(device)
            prediction_scaled = model(b)

        if torch.cuda.is_available(): torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        source_map = prediction_scaled.abs().amax(dim=-1).squeeze(0).cpu().numpy() * Y_TARGET_SCALE
        source_map = np.maximum(source_map, 0.0)
        return create_graph_stc(source_map), elapsed_ms

    def predict_convdip_on_real_evoked(evoked):
        t0 = time.perf_counter()
        try: prediction = convdip_model.predict(evoked)
        except Exception: prediction = convdip_model.predict([evoked])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        stc = prediction[0] if isinstance(prediction, (list, tuple)) else (prediction if hasattr(prediction, "data") else prediction[0])
        source_map = np.max(np.abs(stc.data), axis=1)
        return create_graph_stc(source_map), elapsed_ms

    for condition_name, evoked in evoked_conditions.items():
        cnn_stc, cnn_time = predict_graph_on_real_evoked(evoked, "tikhonov_cnn", tikhonov_cnn_model)
        sparse_stc, sparse_time = predict_graph_on_real_evoked(evoked, "sparse_st_graph", sparse_st_model)
        physics_stc, physics_time = predict_graph_on_real_evoked(evoked, "full_physics_gat", physics_gat_model)
        convdip_stc, convdip_time = predict_convdip_on_real_evoked(evoked)

        all_timing_results.extend([
            {"Seed": current_seed, "Condition": condition_name, "Algorithm": "Tikhonov-CNN", "End-to-end Inference time (ms)": cnn_time},
            {"Seed": current_seed, "Condition": condition_name, "Algorithm": "Sparse ST-Graph", "End-to-end Inference time (ms)": sparse_time},
            {"Seed": current_seed, "Condition": condition_name, "Algorithm": "FullPhysicsGAT", "End-to-end Inference time (ms)": physics_time},
            {"Seed": current_seed, "Condition": condition_name, "Algorithm": "ConvDip", "End-to-end Inference time (ms)": convdip_time}
        ])

        def compute_agreement_metrics(reference_stc, predicted_stc):
            reference_map = maximum_absolute_map(reference_stc)
            predicted_map = maximum_absolute_map(predicted_stc)
            spearman_r, _ = spearmanr(normalize_map(reference_map), normalize_map(predicted_map))
            peak_distance = GEODESIC_MM[peak_index(reference_stc), peak_index(predicted_stc)]
            return {
                "Spearman correlation with dSPM": float(spearman_r),
                "Cosine similarity with dSPM": cosine_similarity(reference_map, predicted_map),
                "Normalized-map MSE vs dSPM": normalized_map_mse(reference_map, predicted_map),
                "Peak distance from dSPM (mm)": float(peak_distance)
            }

        for algorithm_name, predicted_stc in [("Tikhonov-CNN", cnn_stc), ("Sparse ST-Graph", sparse_stc), ("FullPhysicsGAT", physics_stc), ("ConvDip", convdip_stc)]:
            metrics = compute_agreement_metrics(reference_stc=dspm_results[condition_name], predicted_stc=predicted_stc)
            metrics.update({"Seed": current_seed, "Condition": condition_name, "Algorithm": algorithm_name})
            all_real_results.append(metrics)

# ============================================================
# FINAL OUTPUTS AND LOGGING
# ============================================================
pd.DataFrame(all_synthetic_results).to_csv(SYNTHETIC_RESULTS_PATH, index=False)
pd.DataFrame(all_real_results).to_csv(REAL_RESULTS_PATH, index=False)
pd.DataFrame(all_timing_results).to_csv(REAL_TIMING_PATH, index=False)

package_versions = {"esinet": metadata.version("esinet"), "mne": metadata.version("mne"), "torch": torch.__version__, "torch_geometric": metadata.version("torch-geometric"), "numpy": np.__version__}
configuration = {"seeds": SEEDS, "raw_file": raw_file, "packages": package_versions}
with open(CONFIG_PATH, "w", encoding="utf-8") as file: json.dump(configuration, file, ensure_ascii=False, indent=4)

print("\n" + "=" * 75)
print("REAL EEG ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 75)
