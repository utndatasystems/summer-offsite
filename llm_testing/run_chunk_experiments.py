import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from prediction import TokenPredictor
from llm_compressor import LLMCompressor

def run_chunked_experiment(data_path, chunk_size, model_name="Qwen/Qwen2.5-0.5B", context_length=1024, first_n_tokens=10000):
    print(f"\n----- Running Experiment: chunk_size={chunk_size} -----")
    
    # Initialize predictor with chunk_size
    token_predictor = TokenPredictor(
        data_path=data_path,
        model_name=model_name,
        reduce_tokens=True,
        chunk_size=chunk_size,
        first_n_tokens=first_n_tokens
    )
    
    llm_compressor = LLMCompressor()
    data_tokens = token_predictor.get_data_tokens()
    num_chunks = math.ceil(len(data_tokens) / chunk_size)

    total_arithmetic_code_size = 0
    total_bitmap_size = 0
    current_chunk_index = -1

    prompt_tokens = []
    for i in range(len(data_tokens) - 1):
        # Determine the chunk index for the *next* token to be compressed
        next_token_chunk_index = (i + 1) // chunk_size

        # If the next token is in a new chunk, update the active token mask
        if next_token_chunk_index != current_chunk_index:
            _, bitmap_size_bytes = token_predictor.set_active_chunk(next_token_chunk_index)
            total_bitmap_size += bitmap_size_bytes * 8  # convert bytes to bits
            current_chunk_index = next_token_chunk_index

        prompt_tokens.append(data_tokens[i])
        if len(prompt_tokens) > context_length:
            prompt_tokens.pop(0)

        print(f"\rProcessing token {i+1}/{len(data_tokens)}", end='')

        next_token_actual_index = data_tokens[i+1]

        # Get probabilities using the correct, active token mask
        candidate_token_ids, probs_values = token_predictor.get_token_info(prompt_tokens)
        probs_values_np = np.array(probs_values)

        # With the corrected logic, this ValueError should no longer occur,
        # as the token is guaranteed to be in the vocabulary of its own chunk.
        try:
            token_idx_for_compressor = candidate_token_ids.index(next_token_actual_index)
        except ValueError:
            # This is now a critical error, as it indicates a flaw in the logic.
            print(f"\nFATAL: Token {next_token_actual_index} not in the vocabulary for its own chunk {current_chunk_index}. This should not happen.")
            # For safety, we stop compression here as it would lead to data corruption
            break

        llm_compressor.next_token(token_idx_for_compressor, probs_values_np)

    bit_string = llm_compressor.compress()
    total_arithmetic_code_size = len(bit_string)

    final_size = total_arithmetic_code_size + total_bitmap_size
    original_size = len(token_predictor.detokenize(data_tokens)) * 8

    print("\n\n--- Results for Chunk Size {chunk_size} ---")
    print(f"Total Arithmetic Code Size: {total_arithmetic_code_size} bits")
    print(f"Total Bitmap Size: {total_bitmap_size} bits")
    print(f"Final Compressed Size: {final_size} bits")
    print(f"Original Size: {original_size} bits")
    print(f"Overall Compression Ratio: {final_size / original_size * 100:.4f} %")

    return {
        "chunk_size": chunk_size,
        "arithmetic_code_size_bits": total_arithmetic_code_size,
        "bitmap_size_bits": total_bitmap_size,
        "final_size_bits": final_size,
        "compression_ratio_percent": final_size / original_size * 100,
    }

def main():
    parser = argparse.ArgumentParser(description="Run chunked LLM compression experiments and plot results.")
    parser.add_argument("--data_path", type=str, required=True, help="Input text file path.")
    parser.add_argument("--output_plot", type=str, default="chunked_compression_plot.png", help="Output file for the plot.")
    parser.add_argument("--output_json", type=str, default="chunked_compression_results.json", help="Output JSON file to store results.")
    parser.add_argument("--chunk_sizes", type=int, nargs='+', default=[1000, 2000, 5000, 10000], help="List of chunk sizes to test.")
    parser.add_argument("--first_n_tokens", type=int, default=100000, help="Total number of tokens to process from the beginning of the file.")

    args = parser.parse_args()

    # Load existing results if the JSON file exists
    results = []
    if os.path.exists(args.output_json):
        with open(args.output_json, 'r') as f:
            try:
                results = json.load(f)
                print(f"Loaded {len(results)} existing results from {args.output_json}")
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {args.output_json}. Starting with empty results.")

    # Determine which chunk sizes still need to be run
    existing_chunk_sizes = {d['chunk_size'] for d in results}
    chunk_sizes_to_run = [cs for cs in args.chunk_sizes if cs not in existing_chunk_sizes]

    if not chunk_sizes_to_run:
        print("All specified chunk sizes have already been run. Nothing to do.")
    else:
        print(f"Remaining chunk sizes to run: {chunk_sizes_to_run}")

    for chunk_size in chunk_sizes_to_run:
        # Ensure chunk size is not larger than the total tokens to be processed
        if chunk_size > args.first_n_tokens:
            print(f"Skipping chunk size {chunk_size} as it is larger than first_n_tokens ({args.first_n_tokens})")
            continue
        
        result = run_chunked_experiment(
            data_path=args.data_path,
            chunk_size=chunk_size,
            first_n_tokens=args.first_n_tokens
        )
        results.append(result)

        # Save results to JSON after each experiment
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nSaved {len(results)} results to {args.output_json}")

    # Plotting
    if results:
        results.sort(key=lambda d: d['chunk_size'])
        chunk_sizes_str = [str(d['chunk_size']) for d in results]
        arithmetic_sizes = [d['arithmetic_code_size_bits'] for d in results]
        bitmap_sizes = [d['bitmap_size_bits'] for d in results]

        plt.figure(figsize=(12, 8))
        p1 = plt.bar(chunk_sizes_str, arithmetic_sizes, color='#4c72b0', label='Total Arithmetic Code Size')
        p2 = plt.bar(chunk_sizes_str, bitmap_sizes, bottom=arithmetic_sizes, color='#dd8452', label='Total Bitmap Size')

        plt.xlabel('Chunk Size (Number of Tokens)')
        plt.ylabel('Total Size (bits)')
        plt.title(f'Compression Size Breakdown by Chunk Size (First {args.first_n_tokens} Tokens)')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(args.output_plot)
        print(f"\nPlot saved to {args.output_plot}")

if __name__ == "__main__":
    main()
