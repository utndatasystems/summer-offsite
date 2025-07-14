import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from llm_compression.run_compression import run_compression_analysis

def main():
    parser = argparse.ArgumentParser(description="Run LLM compression experiments and plot results.")
    parser.add_argument("--data_path", type=str, default=None,
                        help="The input text file path for LLM inference.")
    parser.add_argument("--text_input", type=str, default=None,
                        help="The direct text input for LLM inference.")
    parser.add_argument("--output_json", type=str, default="compression_results.json",
                        help="Output JSON file to store results.")
    parser.add_argument("--output_plot", type=str, default="compression_plot.png",
                        help="Output file for the plot.")

    args = parser.parse_args()

    if args.data_path is None and args.text_input is None:
        parser.error("Either --data_path or --text_input must be provided.")
    if args.data_path and args.text_input:
        parser.error("Only one of --data_path or --text_input can be provided.")

    # Define experiment settings
    # Example settings, you can customize these
    experiment_settings = [
        # No roaring ranking, just base compression
        {"roaring_ranking_n": None, "reduce_tokens": True},
        # Roaring ranking with different N values
        {"roaring_ranking_n": 5, "reduce_tokens": True},
        {"roaring_ranking_n": 10, "reduce_tokens": True},
        {"roaring_ranking_n": 100, "reduce_tokens": True},
        {"roaring_ranking_n": 1000, "reduce_tokens": True},
        {"roaring_ranking_n": 5000, "reduce_tokens": True},
        {"roaring_ranking_n": 8000, "reduce_tokens": True},
        {"roaring_ranking_n": 9000, "reduce_tokens": True},
        {"roaring_ranking_n": 9998, "reduce_tokens": True},
        {"roaring_ranking_n": 10000, "reduce_tokens": True},
    ]

    results = []
    for i, setting in enumerate(experiment_settings):
        print(f"\n--- Running experiment {i+1}/{len(experiment_settings)}: {setting} ---")
        try:
            result = run_compression_analysis(
                data_path=args.data_path,
                text_input=args.text_input,
                roaring_ranking_n=setting["roaring_ranking_n"],
                reduce_tokens=setting["reduce_tokens"],
                model_name="Qwen/Qwen2.5-0.5B", # You can make this configurable too
                first_n_tokens=10000, # You can make this configurable too
                context_length=512 # You can make this configurable too
            )
            results.append(result)
        except Exception as e:
            print(f"Error running experiment {setting}: {e}")

    # Save results to JSON
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {args.output_json}")

    # Plotting
    if results:
        x = [d['roaring_ranking_n'] if d['roaring_ranking_n'] is not None else 0 for d in results] # Use 0 for None to plot
        # Sort results by roaring_ranking_n for proper plotting
        results.sort(key=lambda d: d['roaring_ranking_n'] if d['roaring_ranking_n'] is not None else -1)
        x_sorted = [d['roaring_ranking_n'] if d['roaring_ranking_n'] is not None else 0 for d in results]

        bitmap_sizes = [d['bitmap_size_bits'] for d in results]
        final_sizes = [d['final_size_bits'] for d in results]

        plt.figure(figsize=(10, 6))
        plt.plot(x_sorted, bitmap_sizes, label='Bitmap Size', color='skyblue', marker='o')
        plt.plot(x_sorted, final_sizes, label='Total Size', color='lightgreen', marker='o')

        # Add horizontal lines from the bitmap_size.ipynb example
        # These values are hardcoded from the example output, you might want to make them dynamic
        plt.axhline(y=52563, color='red', linestyle='--', label='Roaring on distinct token_id (11.7595 % compression ratio)')
        plt.axhline(y=59759, color='orange', linestyle='--', label='No token masking (13.3694 % compression ratio)')

        plt.xlabel('Average probability taken from first N tokens (0 for no ranking)')
        plt.ylabel('Size (bits)')
        plt.title('Roaring Bitmap Storage Size Breakdown')
        plt.legend()
        plt.grid(axis='y')
        plt.xticks(x_sorted) # Ensure all x values are shown as ticks
        plt.tight_layout()
        plt.savefig(args.output_plot)
        print(f"Plot saved to {args.output_plot}")
    else:
        print("No results to plot.")

if __name__ == "__main__":
    main()
