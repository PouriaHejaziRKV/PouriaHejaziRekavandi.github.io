# -*- coding: utf-8 -*-
"""
UNIFIED PIPELINE COMPARISON & OOD BENCHMARK
===========================================
Compares results from:
1) final_pipeline.py (ResidualGraphNet, Tikhonov, dSPM, ConvDip)
2) physics_gat_pipeline.py (PhysicsGAT, SparseGraph, TikhonovCNN, ConvDip)

Outputs:
- ID & OOD comparison plots (PNG)
- Real EEG agreement & inference speed summaries
- Unified CSV summary tables for manuscript/thesis integration
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set visualization style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

def load_data():
    base_dir = Path("/content/drive/MyDrive/Esinet") if os.path.exists("/content/drive/MyDrive") else Path(".")

    final_synthetic_path = base_dir / "final_convdip_graph_comparison/simulation/sample_metrics.csv"
    gat_synthetic_path = base_dir / "physics_gat_final/synthetic_sample_metrics.csv"

    final_real_path = base_dir / "final_convdip_graph_comparison/real_eeg/real_metrics.csv"
    gat_real_path = base_dir / "physics_gat_final/real_eeg_agreement.csv"
    gat_timing_path = base_dir / "physics_gat_final/real_eeg_timing.csv"

    # Check existence
    if not (final_synthetic_path.exists() and gat_synthetic_path.exists()):
        print("[Error] Missing synthetic CSV files. Ensure both pipelines have completed execution.")
        return None, None, None, None, None

    df_final_syn = pd.read_csv(final_synthetic_path)
    df_gat_syn = pd.read_csv(gat_synthetic_path)

    df_final_real = pd.read_csv(final_real_path) if final_real_path.exists() else None
    df_gat_real = pd.read_csv(gat_real_path) if gat_real_path.exists() else None
    df_gat_timing = pd.read_csv(gat_timing_path) if gat_timing_path.exists() else None

    return df_final_syn, df_gat_syn, df_final_real, df_gat_real, df_gat_timing

def plot_synthetic_benchmark(df_final_syn, df_gat_syn):
    output_dir = Path("./comparison_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Standardize Algorithm Names
    df_final_syn["Method"] = df_final_syn["Algorithm"].replace({
        "Proposed Graph": "ResidualGraphNet (Final)",
        "Tikhonov": "Tikhonov Baseline",
        "ConvDip": "ESINet ConvDip",
        "dSPM": "dSPM"
    })

    df_gat_syn["Method"] = df_gat_syn["algorithm"].replace({
        "physics_gat": "Physics-GAT (Proposed)",
        "sparse_graph": "SparseGraphNet",
        "tikhonov_cnn": "Tikhonov-CNN",
        "convdip": "ESINet ConvDip"
    })

    # Prepare In-Distribution (ID) comparison dataframe
    id_final = pd.DataFrame({
        "Method": df_final_syn["Method"],
        "Pipeline": "Final Pipeline",
        "Condition": "ID",
        "Normalized_MSE": df_final_syn["Normalized MSE"],
        "Geodesic_Error_mm": df_final_syn["Geodesic peak error (mm)"],
        "AUC": df_final_syn["ROC AUC"],
        "Inference_ms": df_final_syn["Inference time (ms)"]
    })

    df_gat_id = df_gat_syn[df_gat_syn["condition"] == "ID"]
    id_gat = pd.DataFrame({
        "Method": df_gat_id["Method"],
        "Pipeline": "PHYSICS-GAT Pipeline",
        "Condition": "ID",
        "Normalized_MSE": df_gat_id["map_mse"],
        "Geodesic_Error_mm": df_gat_id["dle_mm"],
        "AUC": df_gat_id["auc"],
        "Inference_ms": df_gat_syn["Inference_ms"] if "Inference_ms" in df_gat_syn.columns else np.nan
    })

    id_combined = pd.concat([id_final, id_gat], ignore_index=True)

    # Figure 1: In-Distribution (ID) Main Benchmark
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.boxplot(data=id_combined, x="Method", y="Normalized_MSE", hue="Pipeline", showfliers=False, ax=axes[0])
    axes[0].set_title("Map Normalized MSE (Lower is Better)")
    axes[0].tick_params(axis='x', rotation=35)

    sns.boxplot(data=id_combined, x="Method", y="Geodesic_Error_mm", hue="Pipeline", showfliers=False, ax=axes[1])
    axes[1].set_title("Geodesic Peak Error (mm) (Lower is Better)")
    axes[1].tick_params(axis='x', rotation=35)

    sns.boxplot(data=id_combined, x="Method", y="AUC", hue="Pipeline", showfliers=False, ax=axes[2])
    axes[2].set_title("ROC AUC Detection Accuracy (Higher is Better)")
    axes[2].tick_params(axis='x', rotation=35)

    plt.suptitle("In-Distribution (ID) Synthetic Performance Benchmark", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "figure1_id_synthetic_benchmark.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: Out-Of-Distribution (OOD) Robustness Benchmark for Physics-GAT Pipeline
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.barplot(data=df_gat_syn, x="condition", y="dle_mm", hue="Method", errorbar="sd", ax=axes[0])
    axes[0].set_title("Peak Geodesic Localization Error (DLE mm)")
    axes[0].set_xlabel("Condition (ID / OOD)")

    sns.barplot(data=df_gat_syn, x="condition", y="map_mse", hue="Method", errorbar="sd", ax=axes[1])
    axes[1].set_title("Normalized Source Map MSE")
    axes[1].set_xlabel("Condition (ID / OOD)")

    sns.barplot(data=df_gat_syn, x="condition", y="auc", hue="Method", errorbar="sd", ax=axes[2])
    axes[2].set_title("ROC AUC Active Source Identification")
    axes[2].set_xlabel("Condition (ID / OOD)")

    plt.suptitle("PHYSICS-GAT Robustness under Out-of-Distribution (OOD) Shifts", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "figure2_ood_robustness_benchmark.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save Unified Summary Table
    # Drop Inference_ms since we have string and numeric mixes
    summary_cols = ["Normalized_MSE", "Geodesic_Error_mm", "AUC"]
    id_combined.groupby(["Pipeline", "Method"])[summary_cols].agg(['mean', 'std']).round(4).to_csv(output_dir / "unified_synthetic_summary.csv")
    print(f"[Success] Synthetic comparison plots and CSV summary saved to: {output_dir}")

def summarize_real_eeg(df_final_real, df_gat_real, df_gat_timing):
    output_dir = Path("./comparison_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("                      REAL EEG ASSESSMENT SUMMARY")
    print("="*80)

    if df_final_real is not None:
        print("\n--- Final Pipeline (dSPM Agreement & Peak Distance) ---")
        summary_final = df_final_real.groupby("Algorithm")[
            ["Spearman agreement with dSPM", "Peak distance from dSPM (mm)", "Inference time (ms)"]
        ].mean(numeric_only=True).round(4)
        print(summary_final)

    if df_gat_real is not None:
        print("\n--- PHYSICS-GAT Pipeline (dSPM Agreement & Geodesic Distance) ---")
        summary_gat = df_gat_real.groupby("algorithm")[
            ["spearman", "cosine", "map_mse", "peak_geodesic_mm"]
        ].mean(numeric_only=True).round(4)
        print(summary_gat)

    if df_gat_timing is not None:
        print("\n--- Inference Time Latency (ms) Across Algorithms ---")
        timing_summary = df_gat_timing.groupby("algorithm")["total_ms"].agg(["mean", "std"]).round(2)
        print(timing_summary)

if __name__ == "__main__":
    df_final_syn, df_gat_syn, df_final_real, df_gat_real, df_gat_timing = load_data()
    if df_final_syn is not None and df_gat_syn is not None:
        plot_synthetic_benchmark(df_final_syn, df_gat_syn)
        summarize_real_eeg(df_final_real, df_gat_real, df_gat_timing)
