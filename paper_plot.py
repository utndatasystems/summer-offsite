import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import defaultdict
import pandas as pd
import json

# --------------------------------------------------------
# (Optional LaTeX config - commented out)
# If LaTeX rendering is needed, uncomment and ensure LaTeX is installed.
# matplotlib.rcParams.update({
#   "pgf.texsystem": "pdflatex",
#   'font.family': 'serif',
#   'text.usetex': True,
#   'pgf.rcfonts': False,
# })
# --------------------------------------------------------

def load_model_results(file_path, selected_datasets=None, selected_n=None):
    """
    Loads model JSON results and flattens into a dictionary of:
        dataset → model → compression/decompression data.
    
    Keeps only the **highest batch size** entry for each dataset+n+model.
    Filters results by dataset and n if specified.
    """
    with open(file_path, "r") as f:
        results = json.load(f)

    # Temporary store to ensure we keep only the largest batch for each dataset/model/n
    temp_store = defaultdict(lambda: defaultdict(dict))  # dataset → model → best entry

    for key, value in results.items():
        dataset_name, model_info = key.split(":", 1)
        parts = model_info.split("|")

        model_name = parts[0]
        n_value = None
        batch_value = None

        # Extract n= and batch= values from the model info
        for p in parts:
            if p.startswith("n="):
                n_value = int(p.split("=")[1])
            elif p.startswith("batch="):
                batch_value = int(p.split("=")[1])

        # Filter datasets/n values if user requested
        if selected_datasets and dataset_name not in selected_datasets:
            continue
        if selected_n and n_value not in selected_n:
            continue

        comp = value.get("compression", {})
        decomp = value.get("decompression", {})

        # Keep only the largest batch for this dataset/model
        if model_name not in temp_store[dataset_name] or batch_value > temp_store[dataset_name][model_name]["batch"]:
            temp_store[dataset_name][model_name] = {
                "batch": batch_value,
                "original_size_bits": comp.get("original_size_bits", 0),
                "ac_bits": comp.get("arithmetic_code_size_bits", 0),
                "bitmap_bits": comp.get("bitmap_size_bits", 0),
                "compressed_size_bits": comp.get("final_size_bits", 0),
                "compression_time": comp.get("total_compression_time_sec", 0),
                "decompression_time": decomp.get("decompression_time_sec", 0)
            }

    # Convert to final format (remove batch value)
    datasets = defaultdict(dict)
    for dataset_name, models in temp_store.items():
        for model_name, info in models.items():
            datasets[dataset_name][model_name] = {
                k: v for k, v in info.items() if k != "batch"
            }
    return datasets


def load_baseline_csv(file_path, dataset_name, selected_compressors=None):
    """
    Reads baseline CSV into dataset → compressor dict.

    Uses the `Uncompressed` row to determine `original_size_bits` for all compressors.
    Filters compressors if a subset is specified.
    """
    df = pd.read_csv(file_path, skip_blank_lines=True)

    # Extract original size from the 'Uncompressed' row
    uncompressed_row = df[df["Compressor"].str.lower() == "uncompressed"]
    if uncompressed_row.empty:
        raise ValueError("CSV must contain an 'Uncompressed' row to determine original size.")
    original_size_bits = int(uncompressed_row.iloc[0]["Size_bytes"]) * 8

    # Filter compressors if user specified a subset
    if selected_compressors:
        df = df[df["Compressor"].isin(selected_compressors)]

    baseline_dict = defaultdict(dict)

    # Store compressor data
    for _, row in df.iterrows():
        if row["Compressor"].lower() == "uncompressed":
            continue  # Skip storing uncompressed in results
        baseline_dict[dataset_name][row["Compressor"]] = {
            "original_size_bits": original_size_bits,
            "ac_bits": 0,  # Baseline compressors have no AC/Bitmap breakdown
            "bitmap_bits": 0,
            "compressed_size_bits": int(row["Size_bytes"]) * 8,
            "compression_time": row.get("Compression_time_s", 0) or 0,
            "decompression_time": row.get("Decompression_time_s", 0) or 0
        }

    return baseline_dict


def merge_results(model_data, baseline_data):
    """
    Merge model results and baseline compressors into one dictionary.
    If both contain the same dataset, entries are combined.
    """
    merged = defaultdict(dict)
    datasets = set(model_data.keys()) | set(baseline_data.keys())
    for ds in datasets:
        merged[ds].update(baseline_data.get(ds, {}))
        merged[ds].update(model_data.get(ds, {}))
    return merged


def results_plot_1(datasets, dataset_order):
    """
    Plot compression ratio breakdown:
    - Model compressors (with AC + Bitmap) as stacked bars.
    - Baseline compressors (no AC/Bitmap split) as single bars.
    """
    fig, axes = plt.subplots(1, len(dataset_order), figsize=(10, 6))

    if len(dataset_order) == 1:
        axes = [axes]

    for idx, dataset in enumerate(dataset_order):
        if dataset not in datasets:
            continue
        comp_dict = datasets[dataset]
        
        ax = axes[idx]
        ax.set_title(f"Compression Ratio Breakdown - {dataset}")

        models = list(comp_dict.keys())
        x = np.arange(len(models))
        bar_width = 0.6

        # Store values
        ac_ratios, bitmap_ratios, total_ratios, is_model_type = [], [], [], []
        for comp_name, m in comp_dict.items():
            orig = m["original_size_bits"]
            comp_size = m["compressed_size_bits"]
            total_ratios.append((comp_size / orig) * 100 if orig else 0)

            if m["ac_bits"] > 0 or m["bitmap_bits"] > 0:
                # Model-type compressor
                ac_ratios.append((m["ac_bits"] / orig) * 100 if orig else 0)
                bitmap_ratios.append((m["bitmap_bits"] / orig) * 100 if orig else 0)
                is_model_type.append(True)
            else:
                # Baseline compressor
                ac_ratios.append(0)
                bitmap_ratios.append(0)
                is_model_type.append(False)

        # Labels to avoid duplicate legend entries
        ac_label_added = bitmap_label_added = baseline_label_added = False

        # Plot bars
        for i, comp_name in enumerate(models):
            if is_model_type[i]:
                ax.bar(
                    i, ac_ratios[i], width=bar_width, color="tab:blue", alpha=0.7,
                    label="AC%" if not ac_label_added else None
                )
                ac_label_added = True

                ax.bar(
                    i, bitmap_ratios[i], width=bar_width, bottom=ac_ratios[i], color="tab:orange", alpha=0.7,
                    label="Bitmap%" if not bitmap_label_added else None
                )
                bitmap_label_added = True
            else:
                ax.bar(
                    i, total_ratios[i], width=bar_width, color="gray", alpha=0.7,
                    label="Baseline" if not baseline_label_added else None
                )
                baseline_label_added = True

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_ylabel("Compression Ratio (%)")
        ax.legend()

    plt.tight_layout()
    plt.savefig("compression_ratio_stack_plot.png")
    # fig.savefig(
    #     f"compression_ratio_stack_plot.pdf",
    #     format="pdf",
    #     bbox_inches="tight",
    #     pad_inches=0.01,
    #     dpi=300
    # )
    plt.close()
    print("Plots saved as compression_ratio_stack_plot.png")


def results_plot_2(datasets, dataset_order):
    """
    Plot compression & decompression speed vs compression ratio:
    - X-axis: Compression Ratio (%)
    - Y-axis: KB/s (log scale)
    - Color: Compressor model
    """
    # Manual color map for compressors
    color_map = {
        "zip": "gray",
        "gzip": "dimgray",
        "zstd -3": "lightgray",
        "zstd -1": "silver",
        "zstd -19": "darkgray",
        "zstd --ultra -22": "black",
        "xz -9e (LZMA2)": "gainsboro",
        "Qwen/Qwen2.5-0.5B": "tab:blue",
        "Qwen/Qwen2.5-1.5B": "tab:orange",
        "Qwen/Qwen2.5-7B": "tab:green",
        "Qwen/Qwen2.5-14B": "tab:red",
        "Qwen/Qwen3-0.6B": "tab:purple",
        "Qwen/Qwen3-1.7B": "tab:brown",
        "Qwen/Qwen3-7B": "tab:pink",
        "Qwen/Qwen3-14B": "tab:cyan",
    }

    # ===== Compression Speed Plot =====
    fig_c, axes_c = plt.subplots(1, len(dataset_order), figsize=(10, 6))
    if len(dataset_order) == 1:
        axes_c = [axes_c]

    for idx, dataset in enumerate(dataset_order):
        if dataset not in datasets:
            continue
        comp_dict = datasets[dataset]
        
        ax_c = axes_c[idx]
        ax_c.set_title(f"Compression Speed - {dataset}")

        for comp_name, m in comp_dict.items():
            orig_bits = m["original_size_bits"]
            comp_bits = m["compressed_size_bits"]
            ratio_percent = (comp_bits / orig_bits) * 100 if orig_bits else 0
            speed_KBps = (orig_bits / 8 / 1e3) / m["compression_time"] if m["compression_time"] > 0 else 0
            color = color_map.get(comp_name, "tab:red")
            ax_c.scatter(ratio_percent, speed_KBps, color=color, label=comp_name)

        ax_c.set_xlabel("Compression Ratio (%)")
        ax_c.set_ylabel("Compression Speed (KB/s)")
        ax_c.set_yscale("log")
        handles, labels = ax_c.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_c.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.savefig("compression_speed_plot.png")
    # plt.savefig(
    #     f"compression_speed_plot.pdf",
    #     format="pdf",
    #     bbox_inches="tight",
    #     pad_inches=0.01,
    #     dpi=300
    # )
    plt.close()

    # ===== Decompression Speed Plot =====
    fig_d, axes_d = plt.subplots(1, len(dataset_order), figsize=(10, 6))
    if len(dataset_order) == 1:
        axes_d = [axes_d]

    for idx, dataset in enumerate(dataset_order):
        if dataset not in datasets:
            continue
        comp_dict = datasets[dataset]
        
        ax_d = axes_d[idx]
        ax_d.set_title(f"Decompression Speed - {dataset}")

        for comp_name, m in comp_dict.items():
            orig_bits = m["original_size_bits"]
            comp_bits = m["compressed_size_bits"]
            ratio_percent = (comp_bits / orig_bits) * 100 if orig_bits else 0
            speed_KBps = (comp_bits / 8 / 1e3) / m["decompression_time"] if m["decompression_time"] > 0 else 0
            color = color_map.get(comp_name, "tab:red")
            ax_d.scatter(ratio_percent, speed_KBps, color=color, label=comp_name)

        ax_d.set_xlabel("Compression Ratio (%)")
        ax_d.set_ylabel("Decompression Speed (KB/s)")
        ax_d.set_yscale("log")
        handles, labels = ax_d.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_d.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.savefig("decompression_speed_plot.png")
    # plt.savefig(
    #     f"decompression_speed_plot.pdf",
    #     format="pdf",
    #     bbox_inches="tight",
    #     pad_inches=0.01,
    #     dpi=300
    # )
    plt.close()

    print("Plots saved as compression_speed_plot.png and decompression_speed_plot.png")


# ===== Example Usage =====
dataset_order = ["text8", "combined_100mb.py"]

model_data = load_model_results(
    "compression_results.json",
    selected_datasets=dataset_order,
    selected_n=[500000]
)

baseline_text8 = load_baseline_csv(
    "text8_baseline.csv",
    dataset_name="text8",
    selected_compressors=["gzip"]
)

pytorrent_data = load_baseline_csv(
    "pytorrent_baseline.csv",
    dataset_name="combined_100mb.py",
    selected_compressors=["gzip"]
)

# Merge results (model + baselines)
final_data = merge_results(model_data, baseline_text8)
final_data = merge_results(final_data, pytorrent_data)

# Plot results
results_plot_1(final_data, dataset_order)
results_plot_2(final_data, dataset_order)
