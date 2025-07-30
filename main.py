import argparse
import json
import os
from llm_testing.global_mask_compressor import run_global_mask_compression

RESULTS_FILE = "compression_results.json"

def load_results():
    """Load previous results from JSON file (if exists)."""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_results(results):
    """Save updated results to JSON file."""
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)

def make_key(args):
    """Generate a unique key for experiment settings."""
    return f"{args.model_name}|ctx={args.context_length}|ret={args.retain_tokens}|n={args.first_n_tokens}|kv={args.use_kv_cache}"

def main():
    # ========================
    # Parse command-line arguments
    # ========================
    parser = argparse.ArgumentParser(description="Run Global Mask Compression Experiment")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset, e.g., ../data/text8")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="Model name")
    parser.add_argument("--context_length", type=int, default=1000, help="Maximum context length")
    parser.add_argument("--retain_tokens", type=int, default=100, help="Tokens retained when context length exceeded (only with KV cache)")
    parser.add_argument("--first_n_tokens", type=int, default=10000, help="Number of tokens to compress")
    parser.add_argument("--use_kv_cache", action="store_true", help="Enable KV cache for compression")
    parser.add_argument("--text_input", type=str, required=False, help="The direct text input for LLM inference.")
    parser.add_argument("--reduce_tokens", type=bool, default=True, help="Whether to restrict the token space to distinct tokens in the input data.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for LLM inference")
    args = parser.parse_args()

    # ========================
    # Check if experiment already exists
    # ========================
    results_db = load_results()
    exp_key = make_key(args)

    # if exp_key in results_db:
    #     print(f"\n⚠️ Experiment already exists for {exp_key}, skipping run.")
    #     print(f"Stored Results: {results_db[exp_key]}")
    #     return

    # ========================
    # Print experiment settings
    # ========================
    print(f"\nRunning compression with parameters:")
    print(f"  Data path        : {args.data_path}")
    print(f"  Model            : {args.model_name}")
    print(f"  Context length   : {args.context_length}")
    print(f"  Retain tokens    : {args.retain_tokens}")
    print(f"  First n tokens   : {args.first_n_tokens}")
    print(f"  Use KV cache     : {args.use_kv_cache}")

    # ========================
    # Run compression experiment
    # ========================
    bit_string, bitmask_data, stats = run_global_mask_compression(args)

    # ========================
    # Save results
    # ========================
    results_db[exp_key] = stats
    save_results(results_db)

    # ========================
    # Output compression results
    # ========================
    print("\n\n===== Compression Results =====")
    for k, v in stats.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
