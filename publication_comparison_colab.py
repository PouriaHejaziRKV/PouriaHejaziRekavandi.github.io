from __future__ import annotations
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
import os
import IPython
print("Installing dependencies...")
IPython.get_ipython().system('pip install -q esinet mne pandas seaborn matplotlib scikit-learn torch-geometric')

# %% 1 — IMPORTS AND GOOGLE DRIVE
try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    pass

import glob
import gc
import json
import time
import copy
import random
import warnings
import importlib.metadata as metadata
import psutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import butter, sosfiltfilt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

import mne

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import (
    Dataset,
    DataLoader,
    TensorDataset
)

from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import add_self_loops, coalesce

from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from scipy.optimize import linear_sum_assignment

from esinet import Simulation
from esinet import Net


warnings.filterwarnings("default")
plt.switch_backend("agg")

mne.set_log_level("WARNING")

# ============================================================
# 1. REPRODUCIBILITY AND CONFIGURATION
# ============================================================

SEED = 42

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_global_seed(SEED)

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

# ------------------------------------------------------------
# Source-space and simulation settings
# ------------------------------------------------------------
SMOKE_TEST = False

SOURCE_SPACING = "ico3"
NUM_SIMULATED_SOURCES = 3
TARGET_SNR = 3.0
SOURCE_EXTENTS = (10, 20)

N_TRAIN = 8 if SMOKE_TEST else 40000
N_VALIDATION = 4 if SMOKE_TEST else 1000
N_TEST = 4 if SMOKE_TEST else 1000

SIMULATION_DURATION = 0.30

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
RESAMPLE_FREQUENCY = 100.0
N_TIMES = int(round(SIMULATION_DURATION * RESAMPLE_FREQUENCY)) + 1

NONLOCAL_K = 4
FUNCTIONAL_K = 3
HIDDEN_DIM = 32
HEADS = 4
DROPOUT = 0.15

if HIDDEN_DIM % HEADS != 0:
    raise ValueError("The hidden dimension must be divisible by the number of attention heads.")

# ------------------------------------------------------------
# Saved files
# ------------------------------------------------------------
GRAPH_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_sparse_st_graph_model.pt")
PHYSICS_GAT_PATH = os.path.join(OUTPUT_DIR, "best_full_physics_gat_model.pt")
CONVDIP_MODEL_PATH = os.path.join(OUTPUT_DIR, "convdip_real_subject_model")
TIKHONOV_CNN_PATH = os.path.join(OUTPUT_DIR, "best_tikhonov_cnn_model.pt")

TRAINING_HISTORY_PATH = os.path.join(OUTPUT_DIR, "graph_training_history.csv")
REAL_RESULTS_PATH = os.path.join(OUTPUT_DIR, "real_data_results.csv")
REAL_TIMING_PATH = os.path.join(OUTPUT_DIR, "real_data_inference_times.csv")
CONFIG_PATH = os.path.join(OUTPUT_DIR, "configuration.json")

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
    if len(results) == 0:
        return None
    return results[0]

def find_subjects_directory(root_directory):
    possible_directories = find_all([os.path.join(root_directory, "**", "subjects")])
    possible_directories.append(root_directory)
    for directory in possible_directories:
        sample_surf = os.path.join(directory, "sample", "surf")
        sample_mri = os.path.join(directory, "sample", "mri")
        if os.path.isdir(sample_surf) and os.path.isdir(sample_mri):
            return directory
    return None

raw_file = find_first([
    os.path.join(MNE_ROOT, "**", "sample_audvis_filt-0-40_raw.fif"),
    os.path.join(MNE_ROOT, "**", "sample_audvis_raw.fif")
])

event_file = find_first([
    os.path.join(MNE_ROOT, "**", "sample_audvis_raw-eve.fif"),
    os.path.join(MNE_ROOT, "**", "*audvis*-eve.fif")
])

trans_file = find_first([
    os.path.join(MNE_ROOT, "**", "sample_audvis_raw-trans.fif"),
    os.path.join(MNE_ROOT, "**", "*audvis*-trans.fif"),
    os.path.join(MNE_ROOT, "**", "*-trans.fif")
])

bem_file = find_first([
    os.path.join(MNE_ROOT, "**", "sample-5120-5120-5120-bem-sol.fif"),
    os.path.join(MNE_ROOT, "**", "sample-*-bem-sol.fif"),
    os.path.join(MNE_ROOT, "**", "*-bem-sol.fif")
])

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
    raw=raw, events=events, event_id=event_id,
    tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
    baseline=(EPOCH_TMIN, 0.0), picks=eeg_picks,
    reject={"eeg": 150e-6}, preload=True, proj=True, detrend=1,
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

    if len(edge_set) == 0:
        raise RuntimeError("The cortical graph contains no edges.")
    return np.array(sorted(edge_set), dtype=np.int64).T

edge_index_np = build_cortical_edges(fwd_fixed)

def make_normalized_sparse_adjacency(edge_index, number_of_nodes, device):
    source = edge_index[0]
    target = edge_index[1]
    self_nodes = np.arange(number_of_nodes, dtype=np.int64)
    rows = np.concatenate([source, target, self_nodes])
    columns = np.concatenate([target, source, self_nodes])
    degree = np.bincount(rows, minlength=number_of_nodes).astype(np.float32)
    degree = np.maximum(degree, 1.0)
    values = (1.0 / np.sqrt(degree[rows] * degree[columns])).astype(np.float32)
    indices = torch.tensor(np.vstack([rows, columns]), dtype=torch.long, device=device)
    values_tensor = torch.tensor(values, dtype=torch.float32, device=device)
    adjacency = torch.sparse_coo_tensor(indices=indices, values=values_tensor, size=(number_of_nodes, number_of_nodes), device=device)
    return adjacency.coalesce()

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

    if finite_values.size == 0:
        raise RuntimeError("No finite geodesic distance was computed.")

    distances[~np.isfinite(distances)] = finite_values.max() + 100.0
    return distances

GEODESIC_MM = compute_geodesic_matrix(edge_index_np, source_coordinates_mm)

def make_local_edge_index() -> torch.Tensor:
    first, second = edge_index_np
    edge_index = torch.tensor(
        np.vstack([np.concatenate([first, second]), np.concatenate([second, first])]),
        dtype=torch.long
    )
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_vertices)
    return coalesce(edge_index, num_nodes=num_vertices)

LOCAL_EDGE_INDEX = make_local_edge_index()

def normalized_leadfield_profiles(lead_field: np.ndarray) -> np.ndarray:
    profiles = np.asarray(lead_field.T, dtype=np.float32).copy()
    profiles -= profiles.mean(axis=1, keepdims=True)
    profiles /= np.linalg.norm(profiles, axis=1, keepdims=True) + 1e-8
    return profiles

if not 1 <= NONLOCAL_K < num_vertices:
    raise ValueError("NONLOCAL_K must be between 1 and V-1.")
if not 1 <= FUNCTIONAL_K < num_vertices:
    raise ValueError("FUNCTIONAL_K must be between 1 and V-1.")

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
        if not self.dynamic:
            return self.base_edges, self.static_attributes
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
# 9. CREATE SUBJECT-SPECIFIC TRAINING SIMULATIONS
# ============================================================
simulation_settings = {
    "duration_of_trial": SIMULATION_DURATION,
    "number_of_sources": NUM_SIMULATED_SOURCES,
    "extents": SOURCE_EXTENTS,
    "target_snr": TARGET_SNR
}

def make_simulation(n_samples, seed):
    set_global_seed(seed)
    simulation = Simulation(fwd_fixed, epochs.info.copy(), settings=simulation_settings)
    simulation.simulate(n_samples=n_samples)
    return simulation

sim_train = make_simulation(N_TRAIN, SEED)
sim_validation = make_simulation(N_VALIDATION, SEED + 1)
sim_test = make_simulation(N_TEST, SEED + 2)

# ============================================================
# 10. TRAIN CONVDIP
# ============================================================
set_global_seed(SEED)
convdip_model = Net(fwd_fixed)
convdip_training_start = time.time()
convdip_model.fit(sim_train, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE)
convdip_training_minutes = (time.time() - convdip_training_start) / 60.0

try:
    convdip_model.save(CONVDIP_MODEL_PATH)
except Exception:
    pass

# ============================================================
# 11. DATA EXTRACTION UTILITIES
# ============================================================
def convert_to_list(data):
    if isinstance(data, list): return data
    if isinstance(data, tuple): return list(data)
    return [data[index] for index in range(len(data))]

def extract_array(item):
    if hasattr(item, "get_data"): return np.asarray(item.get_data())
    if hasattr(item, "data"): return np.asarray(item.data)
    return np.asarray(item)

def ensure_two_dimensions(array):
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[0] == 1: array = array[0]
    return array.astype(np.float32, copy=False)

def extract_eeg_collection(data):
    output = []
    for item in convert_to_list(data):
        output.append(ensure_two_dimensions(extract_array(item)))
    return np.stack(output, axis=0)

def extract_source_full(data):
    output = []
    for item in convert_to_list(data):
        source_data = ensure_two_dimensions(extract_array(item))
        output.append(source_data)
    return np.stack(output, axis=0).astype(np.float32)

X_eeg_train = extract_eeg_collection(sim_train.eeg_data)
Y_source_train = extract_source_full(sim_train.source_data)
X_eeg_validation = extract_eeg_collection(sim_validation.eeg_data)
Y_source_validation = extract_source_full(sim_validation.source_data)
X_eeg_test = extract_eeg_collection(sim_test.eeg_data)
Y_source_test = extract_source_full(sim_test.source_data)

assert_finite_array("X_eeg_train", X_eeg_train)
assert_finite_array("Y_source_train", Y_source_train)

# ============================================================
# 12. REGULARIZED TIKHONOV INVERSE
# ============================================================
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
X_source_test = apply_tikhonov_batch(X_eeg_test, K_dagger)

# ============================================================
# 13. TRAIN-ONLY NORMALIZATION
# ============================================================
def robust_scale(array, percentile=99.5):
    value = np.percentile(np.abs(array), percentile)
    return float(max(value, 1e-12))

X_INPUT_SCALE = robust_scale(X_source_train)
Y_TARGET_SCALE = robust_scale(Y_source_train)

X_source_train = np.clip(X_source_train / X_INPUT_SCALE, -10.0, 10.0).astype(np.float32)
X_source_validation = np.clip(X_source_validation / X_INPUT_SCALE, -10.0, 10.0).astype(np.float32)
X_source_test = np.clip(X_source_test / X_INPUT_SCALE, -10.0, 10.0).astype(np.float32)

Y_source_train = np.clip(Y_source_train / Y_TARGET_SCALE, -10.0, 10.0).astype(np.float32)
Y_source_validation = np.clip(Y_source_validation / Y_TARGET_SCALE, -10.0, 10.0).astype(np.float32)
Y_source_test = np.clip(Y_source_test / Y_TARGET_SCALE, -10.0, 10.0).astype(np.float32)

# ============================================================
# 14. NEURAL ARCHITECTURES
# ============================================================

# 14A. TikhonovTemporalCNN
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

# 14B. Sparse Spatio-Temporal Graph Network
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

# 14C. Full Physics GAT
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

    def forward_one(self, initial_source, edge_index, edge_attr):
        features = self.temporal_encoder(initial_source)
        features = self.graph_block_1(features, edge_index, edge_attr)
        features = self.graph_block_2(features, edge_index, edge_attr)
        output = self.temporal_decoder(features)
        return output

    def forward(self, initial_source, edge_indices, edge_attributes):
        if initial_source.ndim != 3: raise ValueError("Expected initial_source with shape [B, V, T].")
        if initial_source.shape[1] != num_vertices: raise ValueError("Unexpected number of source vertices.")
        outputs = []
        for i in range(len(initial_source)):
            ei = edge_indices[i].to(initial_source.device)
            ea = edge_attributes[i].to(initial_source.device)
            outputs.append(self.forward_one(initial_source[i], ei, ea))
        return torch.stack(outputs)

# ============================================================
# 15. LOSS TENSORS
# ============================================================
GEODESIC_TENSOR = torch.tensor(GEODESIC_MM, dtype=torch.float32, device=device)

def graph_training_loss(prediction, target, graph_edges, active_weight, active_threshold_ratio, spatial_weight):
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
    geodesic_terms = []
    for i in range(len(prediction)):
        true_support = torch.where(active_target[i])[0]
        if len(true_support) == 0:
            true_support = target_map[i].argmax().reshape(1)
        distance_to_true_support = GEODESIC_TENSOR[:, true_support].amin(dim=1)
        geodesic_terms.append((probability[i] * distance_to_true_support).sum() / 100.0)
    geodesic_loss = torch.stack(geodesic_terms).mean()

    total = waveform_loss + map_loss + spatial_weight * edge_smoothness + 2e-3 * geodesic_loss
    return total, waveform_loss.detach(), map_loss.detach()

# ============================================================
# 16. DATA LOADERS
# ============================================================
class PhysicsDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = x_data
        self.y = y_data
    def __len__(self): return len(self.x)
    def __getitem__(self, idx):
        source_ts = self.x[idx]
        target = self.y[idx]
        edge_index, edge_attr = DYNAMIC_GRAPH_BUILDER(source_ts)
        return {"x": torch.from_numpy(source_ts), "y": torch.from_numpy(target), "edge_index": edge_index, "edge_attr": edge_attr}

def collate_physics(batch):
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "edge_index": [b["edge_index"] for b in batch],
        "edge_attr": [b["edge_attr"] for b in batch],
        "adjacency": sparse_adjacency # For sparse model
    }

train_dataset = PhysicsDataset(X_source_train, Y_source_train)
validation_dataset = PhysicsDataset(X_source_validation, Y_source_validation)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=0, collate_fn=collate_physics)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=0, collate_fn=collate_physics)

# ============================================================
# 17. TRAIN NEURAL MODELS
# ============================================================
def execute_epoch(model_name, model, loader, optimizer, training):
    if training: model.train()
    else: model.eval()

    total_loss, total_wave, total_map, total_samples = 0.0, 0.0, 0.0, 0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch in loader:
            batch_x = batch["x"].to(device, non_blocking=True)
            batch_y = batch["y"].to(device, non_blocking=True)

            if training: optimizer.zero_grad(set_to_none=True)

            if model_name == "sparse_st_graph": prediction = model(batch_x, batch["adjacency"])
            elif model_name == "tikhonov_cnn": prediction = model(batch_x)
            else: prediction = model(batch_x, batch["edge_index"], batch["edge_attr"])

            loss, wave_l, map_l = graph_training_loss(prediction, batch_y, edge_index_tensor, ACTIVE_WEIGHT, ACTIVE_THRESHOLD_RATIO, SPATIAL_LOSS_WEIGHT)

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

def train_model_loop(model_name, model, save_path):
    print(f"\n{'='*75}\nTRAINING: {model_name.upper()}\n{'='*75}")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE, min_lr=1e-6)

    history = []
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    start_time = time.time()

    last_save_path = save_path.replace("best", "last")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_stats = execute_epoch(model_name, model, train_loader, optimizer, training=True)
        val_stats = execute_epoch(model_name, model, validation_loader, optimizer, training=False)

        validation_loss = val_stats["loss"]
        scheduler.step(validation_loss)
        learning_rate = optimizer.param_groups[0]["lr"]

        history.append({
            "Epoch": epoch, "Training loss": train_stats["loss"], "Validation loss": val_stats["loss"],
            "Training wave": train_stats["wave"], "Validation wave": val_stats["wave"],
            "Training map": train_stats["map"], "Validation map": val_stats["map"], "Learning rate": learning_rate
        })

        improved = validation_loss < best_validation_loss - 1e-7
        checkpoint_data = {
            "model_state_dict": model.state_dict(), "epoch": epoch, "validation_loss": validation_loss,
            "input_scale": X_INPUT_SCALE, "target_scale": Y_TARGET_SCALE,
            "channel_names": epochs.ch_names, "vertices": [fwd_fixed["src"][0]["vertno"], fwd_fixed["src"][1]["vertno"]],
            "history": history
        }

        torch.save(checkpoint_data, last_save_path)

        if improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(checkpoint_data, save_path)
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            print(f"Epoch {epoch:03d} | Train={train_stats['loss']:.6f} | Val={validation_loss:.6f} | LR={learning_rate:.2e}")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Early stopping at epoch", epoch)
            break

    training_minutes = (time.time() - start_time) / 60.0
    print(f"{model_name} training time: {training_minutes:.2f} minutes")

    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded best {model_name} checkpoint from epoch:", checkpoint["epoch"])

    return history, training_minutes

tikhonov_cnn_model = TikhonovTemporalCNN().to(device)
sparse_st_model = SparseSpatioTemporalGraphNet(hidden_dimension=32, dropout=0.20).to(device)
physics_gat_model = FullPhysicsGAT().to(device)

cnn_history, cnn_train_mins = train_model_loop("tikhonov_cnn", tikhonov_cnn_model, TIKHONOV_CNN_PATH)
sparse_history, sparse_train_mins = train_model_loop("sparse_st_graph", sparse_st_model, GRAPH_MODEL_PATH)
physics_history, physics_train_mins = train_model_loop("full_physics_gat", physics_gat_model, PHYSICS_GAT_PATH)

pd.DataFrame(sparse_history).to_csv(TRAINING_HISTORY_PATH, index=False)
pd.DataFrame(physics_history).to_csv(os.path.join(OUTPUT_DIR, "physics_gat_history.csv"), index=False)
pd.DataFrame(cnn_history).to_csv(os.path.join(OUTPUT_DIR, "tikhonov_cnn_history.csv"), index=False)

# SMOKE TEST VERIFICATION
if SMOKE_TEST:
    test_item = train_dataset[0]
    assert_finite_array("test x", test_item["x"])
    assert_finite_array("test y", test_item["y"])
    assert_finite_array("test edge_attr", test_item["edge_attr"])

    with torch.no_grad():
        test_pred = physics_gat_model(test_item["x"][None].to(device), [test_item["edge_index"]], [test_item["edge_attr"]])
    assert_finite_array("test prediction", test_pred)
    print("Smoke test passed.")

# ============================================================
# 18. dSPM CLASSICAL INVERSE ON REAL EEG
# ============================================================
print("\n" + "=" * 75)
print("PHASE 6: COMPUTING REAL EEG dSPM REFERENCE")
print("=" * 75)

noise_covariance = mne.compute_covariance(epochs, tmin=EPOCH_TMIN, tmax=0.0, method=["shrunk", "empirical"], rank=None, verbose=False)
dspm_inverse_operator = mne.minimum_norm.make_inverse_operator(info=epochs.info, forward=fwd_free, noise_cov=noise_covariance, loose=0.2, depth=0.8, fixed=False, rank=None, verbose=False)
lambda2 = 1.0 / TARGET_SNR ** 2

dspm_results = {}
for condition_name, evoked in evoked_conditions.items():
    dspm_results[condition_name] = mne.minimum_norm.apply_inverse(evoked, dspm_inverse_operator, lambda2=lambda2, method="dSPM", pick_ori=None, verbose=False)

# ============================================================
# 19. INFERENCE UTILITIES
# ============================================================
def create_graph_stc(source_map):
    return mne.SourceEstimate(data=source_map[:, np.newaxis], vertices=[fwd_fixed["src"][0]["vertno"], fwd_fixed["src"][1]["vertno"]], tmin=0.0, tstep=1.0, subject="sample")

@torch.no_grad()
def predict_graph_on_real_evoked(evoked, model_name, model):
    eeg_data = np.asarray(evoked.data, dtype=np.float32)
    source_initialization = np.einsum("vc,ct->vt", K_dagger, eeg_data, optimize=True).astype(np.float32)
    source_initialization = np.clip(source_initialization / X_INPUT_SCALE, -10.0, 10.0)
    model_input = torch.from_numpy(source_initialization).unsqueeze(0).to(device)

    if torch.cuda.is_available(): torch.cuda.synchronize()
    start = time.perf_counter()

    if model_name == "sparse_st_graph": prediction_scaled = model(model_input, sparse_adjacency)
    elif model_name == "tikhonov_cnn": prediction_scaled = model(model_input)
    else:
        edge_index, edge_attr = DYNAMIC_GRAPH_BUILDER(source_initialization)
        prediction_scaled = model(model_input, [edge_index], [edge_attr])

    if torch.cuda.is_available(): torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    source_map = prediction_scaled.abs().amax(dim=-1).squeeze(0).cpu().numpy() * Y_TARGET_SCALE
    source_map = np.maximum(source_map, 0.0)

    return create_graph_stc(source_map), elapsed_ms

def normalize_convdip_prediction(prediction):
    if isinstance(prediction, (list, tuple)): return prediction[0]
    if hasattr(prediction, "data"): return prediction
    return prediction[0]

def predict_convdip_on_real_evoked(evoked):
    start = time.perf_counter()
    try: prediction = convdip_model.predict(evoked)
    except Exception: prediction = convdip_model.predict([evoked])
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    stc = normalize_convdip_prediction(prediction)
    source_map = np.max(np.abs(stc.data), axis=1)
    return create_graph_stc(source_map), elapsed_ms

# ============================================================
# 20. RUN ALL NEURAL INVERSE METHODS ON REAL EEG
# ============================================================
print("\n" + "=" * 75)
print("PHASE 7: NEURAL INVERSE ON REAL EEG")
print("=" * 75)

cnn_results, sparse_results, physics_results, convdip_results = {}, {}, {}, {}
timing_rows = []

first_evoked = next(iter(evoked_conditions.values()))
_ = predict_graph_on_real_evoked(first_evoked, "tikhonov_cnn", tikhonov_cnn_model)
_ = predict_graph_on_real_evoked(first_evoked, "sparse_st_graph", sparse_st_model)
_ = predict_graph_on_real_evoked(first_evoked, "full_physics_gat", physics_gat_model)
try: _ = predict_convdip_on_real_evoked(first_evoked)
except Exception: pass

for condition_name, evoked in evoked_conditions.items():
    cnn_stc, cnn_time = predict_graph_on_real_evoked(evoked, "tikhonov_cnn", tikhonov_cnn_model)
    sparse_stc, sparse_time = predict_graph_on_real_evoked(evoked, "sparse_st_graph", sparse_st_model)
    physics_stc, physics_time = predict_graph_on_real_evoked(evoked, "full_physics_gat", physics_gat_model)
    convdip_stc, convdip_time = predict_convdip_on_real_evoked(evoked)

    cnn_results[condition_name] = cnn_stc
    sparse_results[condition_name] = sparse_stc
    physics_results[condition_name] = physics_stc
    convdip_results[condition_name] = convdip_stc

    timing_rows.extend([
        {"Condition": condition_name, "Algorithm": "Tikhonov-CNN", "Inference time (ms)": cnn_time},
        {"Condition": condition_name, "Algorithm": "Sparse ST-Graph", "Inference time (ms)": sparse_time},
        {"Condition": condition_name, "Algorithm": "FullPhysicsGAT", "Inference time (ms)": physics_time},
        {"Condition": condition_name, "Algorithm": "ConvDip", "Inference time (ms)": convdip_time}
    ])

timing_df = pd.DataFrame(timing_rows)
timing_df.to_csv(REAL_TIMING_PATH, index=False)

# ============================================================
# 21. REAL-DATA AGREEMENT METRICS
# ============================================================
def maximum_absolute_map(stc): return np.max(np.abs(stc.data), axis=1)

def normalize_map(source_map):
    source_map = np.asarray(source_map, dtype=np.float64)
    return source_map / (np.max(np.abs(source_map)) + 1e-12)

def normalized_map_mse(first_map, second_map): return float(np.mean((normalize_map(first_map) - normalize_map(second_map)) ** 2))

def cosine_similarity(first_map, second_map): return float(1.0 - cosine(normalize_map(first_map), normalize_map(second_map)))

def peak_index(stc): return int(np.argmax(maximum_absolute_map(stc)))

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

real_result_rows = []
for condition_name in evoked_conditions:
    for algorithm_name, predicted_stc in [
        ("Tikhonov-CNN", cnn_results[condition_name]),
        ("Sparse ST-Graph", sparse_results[condition_name]),
        ("FullPhysicsGAT", physics_results[condition_name]),
        ("ConvDip", convdip_results[condition_name])
    ]:
        metrics = compute_agreement_metrics(reference_stc=dspm_results[condition_name], predicted_stc=predicted_stc)
        metrics.update({"Condition": condition_name, "Algorithm": algorithm_name})
        real_result_rows.append(metrics)

real_results_df = pd.DataFrame(real_result_rows)
real_results_df.to_csv(REAL_RESULTS_PATH, index=False)
print("\nReal-data agreement with dSPM:")
print(real_results_df.round(4))

# ============================================================
# 26. SAVE CONFIGURATION
# ============================================================
package_versions = {
    "esinet": metadata.version("esinet"),
    "mne": metadata.version("mne"),
    "torch": torch.__version__,
    "torch_geometric": metadata.version("torch-geometric"),
    "numpy": np.__version__
}

configuration = {
    "seed": SEED, "raw_file": raw_file, "packages": package_versions,
    "convdip_training_minutes": convdip_training_minutes, "sparse_st_training_minutes": sparse_train_mins,
    "physics_gat_training_minutes": physics_train_mins, "tikhonov_cnn_training_minutes": cnn_train_mins
}
with open(CONFIG_PATH, "w", encoding="utf-8") as file: json.dump(configuration, file, ensure_ascii=False, indent=4)

print("\n" + "=" * 75)
print("REAL EEG ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 75)
