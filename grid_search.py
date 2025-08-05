import os
from pathlib import Path
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


BATCH_SIZES = [4, 16, 64, 256]
CONTEXT_LENGTHS = [128, 256, 512, 1024]
NUM_TOKENS = 1_000_000


def plot(metric, title, label, figname):
    with open("compression_results.json") as f:
        results = json.load(f)

    rows = []
    for _, result in results.items():
        if result["compression"]["args"]["batch_size"] not in BATCH_SIZES:
            continue
        if result["compression"]["args"]["context_length"] not in CONTEXT_LENGTHS:
            continue

        batch_size = int(result["compression"]["args"]["batch_size"])
        context_length = int(result["compression"]["args"]["context_length"])
        throughput = metric(result)

        rows.append({
            "context_len": context_length,
            "batch_size": batch_size,
            "throughput": throughput
        })

    df = (
        pd.DataFrame(rows)
        .groupby(["context_len", "batch_size"])["throughput"]
        .mean()
        .unstack()
    )

    plt.figure(figsize=(8, 5))
    sns.heatmap(df, annot=True, fmt=".2f", linewidths=.5, cbar_kws={"label": label})
    plt.xlabel("Batch size")
    plt.ylabel("Context length")
    plt.title(title)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/{figname}.png")


if __name__ == "__main__":
    for batch_size in BATCH_SIZES:
        for context_length in CONTEXT_LENGTHS:
            os.system(
                f"python main.py --input_path data/text8 --mode compress --batch_size {batch_size} --context_length {context_length} --first_n_tokens {NUM_TOKENS} --use_kv_cache"
            )

    plot(
        lambda stats: float(stats["compression"]["inference_throughput_kibibytes_per_sec"]),
        "Processed KiB/s",
        "KiB/s",
        "inference_throughput_heatmap"
    )
    plot(
        lambda stats: float(stats["compression"]["compression_factor"]),
        "Compression Factor",
        "Factor",
        "compression_factor_heatmap"
    )
    plot(
        lambda stats: float(stats["compression"]["pure_compression_factor"]),
        "Pure Compression Factor",
        "Factor",
        "pure_compression_factor_heatmap"
    )