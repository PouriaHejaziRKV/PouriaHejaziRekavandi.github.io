import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_synthetic_comparison():
    import os
    base_dir = "/content/drive/MyDrive/Esinet" if os.path.exists("/content/drive/MyDrive") else "."

    final_synthetic = Path(f"{base_dir}/final_convdip_graph_comparison/simulation/sample_metrics.csv")
    gat_synthetic = Path(f"{base_dir}/physics_gat_final/synthetic_sample_metrics.csv")

    if not final_synthetic.exists() or not gat_synthetic.exists():
        print("Missing synthetic metric CSVs. Run pipelines first.")
        return

    df_final = pd.read_csv(final_synthetic)
    df_gat = pd.read_csv(gat_synthetic)

    # Rename algorithms in final to clarify
    df_final["Algorithm"] = df_final["Algorithm"].replace({
        "Proposed Graph": "Final_Graph",
        "Tikhonov": "Final_Tikhonov",
        "ConvDip": "Final_ConvDip"
    })

    df_gat["Algorithm"] = df_gat["algorithm"].replace({
        "physics_gat": "GAT_Physics",
        "sparse_graph": "GAT_SparseGraph",
        "tikhonov_cnn": "GAT_TikhonovCNN",
        "convdip": "GAT_ConvDip"
    })

    # Map metrics to a common name for comparison
    df_final_renamed = pd.DataFrame()
    df_final_renamed["Algorithm"] = df_final["Algorithm"]
    df_final_renamed["Pipeline"] = "Final Pipeline"
    df_final_renamed["MSE"] = df_final["MSE"]
    df_final_renamed["Geodesic_Error_mm"] = df_final["Geodesic peak error (mm)"]

    df_gat_renamed = pd.DataFrame()
    # Filter GAT ID condition to match the Final Pipeline which tests on a single synthetic set
    df_gat_id = df_gat[df_gat["condition"] == "ID"]
    df_gat_renamed["Algorithm"] = df_gat_id["Algorithm"]
    df_gat_renamed["Pipeline"] = "PHYSICS-GAT Pipeline"
    df_gat_renamed["MSE"] = df_gat_id["waveform_mse"] # or map_mse
    df_gat_renamed["Geodesic_Error_mm"] = df_gat_id["dle_mm"]

    combined = pd.concat([df_final_renamed, df_gat_renamed]).reset_index(drop=True)

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=combined, x="Algorithm", y="MSE", hue="Pipeline")
    plt.xticks(rotation=45, ha='right')
    plt.title("Synthetic MSE Comparison")

    plt.subplot(1, 2, 2)
    sns.boxplot(data=combined, x="Algorithm", y="Geodesic_Error_mm", hue="Pipeline")
    plt.xticks(rotation=45, ha='right')
    plt.title("Synthetic Geodesic Peak Error (mm)")

    plt.tight_layout()
    plt.savefig("synthetic_comparison_plot.png")
    print("Saved synthetic_comparison_plot.png")

def summarize_real_eeg():
    import os
    base_dir = "/content/drive/MyDrive/Esinet" if os.path.exists("/content/drive/MyDrive") else "."

    final_real = Path(f"{base_dir}/final_convdip_graph_comparison/real_eeg/real_metrics.csv")
    gat_real = Path(f"{base_dir}/physics_gat_final/real_eeg_agreement.csv")

    if not final_real.exists() or not gat_real.exists():
        print("Missing real EEG metric CSVs. Run pipelines first.")
        return

    df_final = pd.read_csv(final_real)
    df_gat = pd.read_csv(gat_real)

    print("\n--- Real EEG dSPM Agreement Summary ---")
    print("\nFinal Pipeline:")
    print(df_final.groupby("Algorithm")[["Spearman agreement with dSPM", "Peak distance from dSPM (mm)"]].mean())

    print("\nPHYSICS-GAT Pipeline:")
    print(df_gat.groupby("algorithm")[["spearman", "peak_geodesic_mm"]].mean())

if __name__ == "__main__":
    plot_synthetic_comparison()
    summarize_real_eeg()