import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr

def compute_cosine_sim_batch(e1, e2):
    e1_norm = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
    e2_norm = e2 / np.linalg.norm(e2, axis=1, keepdims=True)
    return np.sum(e1_norm * e2_norm, axis=1)

def compute_l1_sim_paper_batch(e1, e2):
    dot_product = np.sum(e1 * e2, axis=1)
    norm_l1_e1 = np.linalg.norm(e1, ord=1, axis=1)
    norm_l1_e2 = np.linalg.norm(e2, ord=1, axis=1)
    return dot_product / (norm_l1_e1 * norm_l1_e2)

def compute_l1_norm_batch(e1, e2):
    return np.linalg.norm(e1 - e2, ord=1, axis=1)

def compute_l2_norm_batch(e1, e2):
    return np.linalg.norm(e1 - e2, ord=2, axis=1)

path_pca = "stsb/pca_data/test-00000-of-00001_embedded_pca110.parquet"
df_pca = pd.read_parquet(path_pca)

path_wpca = "stsb/weighted_pca_data/test-00000-of-00001_embedded_wpca110.parquet"
df_wpca = pd.read_parquet(path_wpca)

assert len(df_pca) == len(df_wpca), "Data length mismatch！"

y_true = df_pca['score'].values / 5.0

emb1_orig = np.stack(df_pca['embedding1'].values).astype(np.float32)
emb2_orig = np.stack(df_pca['embedding2'].values).astype(np.float32)

emb1_pca = np.stack(df_pca['embedding1_pca'].values).astype(np.float32)
emb2_pca = np.stack(df_pca['embedding2_pca'].values).astype(np.float32)

emb1_wpca = np.stack(df_wpca['embedding1_wpca'].values).astype(np.float32)
emb2_wpca = np.stack(df_wpca['embedding2_wpca'].values).astype(np.float32)


def run_experiment(name, e1, e2):
    results = {}
    latencies = {}
    metrics = [
        ("Cosine Similarity", compute_cosine_sim_batch),
        ("L1 Similarity", compute_l1_sim_paper_batch),
        ("L1 Norm", compute_l1_norm_batch),
        ("L2 Norm", compute_l2_norm_batch)
    ]
    scaler = MinMaxScaler()

    for metric_name, func in metrics:
        iters = 100
        start = time.time()
        for _ in range(iters):
            raw_values = func(e1, e2)
        end = time.time()

        avg_latency = ((end - start) / iters) / len(e1) * 1000
        latencies[metric_name] = avg_latency

        vals_2d = raw_values.reshape(-1, 1)
        if "Norm" in metric_name:
            scaled_vals = 1 - scaler.fit_transform(vals_2d).flatten()
        else:
            scaled_vals = scaler.fit_transform(vals_2d).flatten()

        mae = np.mean(np.abs(scaled_vals - y_true))
        corr, _ = pearsonr(scaled_vals, y_true)

        results[metric_name] = {"MAE": mae, "Pearson": corr}

    return results, latencies


print("Running 3072-dimensional Baseline test...")
res_orig, lat_orig = run_experiment("Original", emb1_orig, emb2_orig)

print("Running 110-dimensional Standard PCA test...")
res_pca, lat_pca = run_experiment("PCA-110", emb1_pca, emb2_pca)

print("Running 110-dimensional Weighted PCA test...")
res_wpca, lat_wpca = run_experiment("Weighted-PCA-110", emb1_wpca, emb2_wpca)

print("\n" + "=" * 80)
print("Table 2: Latency Comparison (ms per pair) - Speed Comparison")
print("=" * 80)
print(f"{'Metric':<20} | {'Original(3072)':<15} | {'Std-PCA(110)':<15} | {'W-PCA(110)':<15} | {'Speedup':<10}")
print("-" * 80)
for m in lat_orig.keys():
    speedup = lat_orig[m] / lat_wpca[m]
    print(f"{m:<20} | {lat_orig[m]:<15.6f} | {lat_pca[m]:<15.6f} | {lat_wpca[m]:<15.6f} | {speedup:<10.2f}x")

print("\n" + "=" * 85)
print("Table 3: Accuracy Comparison (MAE) - Error Comparison (Lower is better)")
print("=" * 85)
print(f"{'Metric':<20} | {'Orig MAE':<10} | {'Std-PCA MAE':<12} | {'W-PCA MAE':<12} | {'Improv vs PCA':<15}")
print("-" * 85)
for m in res_orig.keys():
    print(f"{m:<20} | {res_orig[m]['MAE']:<10.4f} | {res_pca[m]['MAE']:<12.4f} | {res_wpca[m]['MAE']:<12.4f} | {res_wpca[m]['MAE'] - res_pca[m]['MAE']:<15.4f}")

print("\n" + "=" * 85)
print("Table 4: Correlation Comparison (Pearson) - Correlation Comparison (Higher is better)")
print("=" * 85)
print(f"{'Metric':<20} | {'Orig Corr':<10} | {'Std-PCA Corr':<12} | {'W-PCA Corr':<12} | {'Improv vs PCA':<15}")
print("-" * 85)
for m in res_orig.keys():
    print(f"{m:<20} | {res_orig[m]['Pearson']:<10.4f} | {res_pca[m]['Pearson']:<12.4f} | {res_wpca[m]['Pearson']:<12.4f} | {res_wpca[m]['Pearson'] - res_pca[m]['Pearson']:<15.4f}")
print("=" * 85)

OUTPUT_IMG_DIR = "stsb/plots"
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# Set a unified visual style
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
# Color scheme: Orange (Original), Blue (Standard PCA), Green (Weighted PCA)
color_palette = ["#faa76c", "#5c9ebc", "#2ca02c"]

plot_data = []
for m in res_orig.keys():
    plot_data.append({"Metric": m, "Model": "Original (3072d)", "MAE": res_orig[m]["MAE"], "Pearson": res_orig[m]["Pearson"], "Latency": lat_orig[m]})
    plot_data.append({"Metric": m, "Model": "Std-PCA (110d)", "MAE": res_pca[m]["MAE"], "Pearson": res_pca[m]["Pearson"], "Latency": lat_pca[m]})
    plot_data.append({"Metric": m, "Model": "W-PCA (110d) [Ours]", "MAE": res_wpca[m]["MAE"], "Pearson": res_wpca[m]["Pearson"], "Latency": lat_wpca[m]})
df_plot = pd.DataFrame(plot_data)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_plot, x="Metric", y="MAE", hue="Model", palette=color_palette)
plt.title("Mean Absolute Error (MAE) Comparison \n(Lower is Better)", fontsize=14, fontweight='bold')
plt.ylabel("MAE", fontsize=12)
plt.xlabel("Similarity/Distance Metric", fontsize=12)
plt.legend(title="Model Configuration", loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_IMG_DIR, "eval_mae_comparison.png"), dpi=300)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_plot, x="Metric", y="Pearson", hue="Model", palette=color_palette)
plt.title("Pearson Correlation Comparison \n(Higher is Better)", fontsize=14, fontweight='bold')
plt.ylabel("Pearson Correlation Coefficient", fontsize=12)
plt.xlabel("Similarity/Distance Metric", fontsize=12)
plt.legend(title="Model Configuration", loc='lower left')
# Add two auxiliary lines to indicate the ceiling
plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_IMG_DIR, "eval_pearson_comparison.png"), dpi=300)

plt.figure(figsize=(10, 6))
# Select Cosine Similarity as the main representative for plotting
df_tradeoff = df_plot[df_plot["Metric"] == "Cosine Similarity"]

sns.scatterplot(data=df_tradeoff, x="Latency", y="Pearson", hue="Model",
                palette=color_palette, s=200, edgecolor='black', zorder=5)

# Draw connecting lines to show the shift of the Pareto frontier
latencies = df_tradeoff["Latency"].values
pearsons = df_tradeoff["Pearson"].values
# Connect Std-PCA to W-PCA with a line
plt.plot([latencies[1], latencies[2]], [pearsons[1], pearsons[2]], color='gray', linestyle='--', zorder=1)
# Connect W-PCA to Original with a line
plt.plot([latencies[2], latencies[0]], [pearsons[2], pearsons[0]], color='gray', linestyle=':', zorder=1)

# Add text annotations
plt.annotate('~50x Speedup', xy=((latencies[0]+latencies[2])/2, (pearsons[0]+pearsons[2])/2),
             xytext=(0, 20), textcoords='offset points', ha='center', color='gray')
plt.annotate('Free Accuracy Boost', xy=(latencies[2], (pearsons[1]+pearsons[2])/2),
             xytext=(50, 0), textcoords='offset points', ha='left', color='#2ca02c', fontweight='bold', arrowprops=dict(arrowstyle="->", color='#2ca02c'))

plt.xscale('log') # Use logarithmic coordinate system because 0.015 and 0.0003 differ significantly
plt.title("Accuracy vs. Latency Trade-off (Cosine Similarity)\n(Top-Left is the Optimal Zone)", fontsize=14, fontweight='bold')
plt.xlabel("Latency per Pair (ms) [Log Scale]", fontsize=12)
plt.ylabel("Pearson Correlation", fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_IMG_DIR, "eval_pareto_tradeoff.png"), dpi=300)

