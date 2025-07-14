import argparse
import numpy as np
import torch
from prediction import TokenPredictor
from llm_compressor import LLMCompressor



def run_compression_analysis(data_path=None, text_input=None, roaring_ranking_n=None, model_name="Qwen/Qwen2.5-0.5B", reduce_tokens=True, first_n_tokens=10000, context_length=512):
    if data_path is None and text_input is None:
        raise ValueError("Either data_path or text_input must be provided.")
    if data_path and text_input:
        raise ValueError("Only one of data_path or text_input can be provided.")

    print(f"Input source: {'File' if data_path else 'Text'}")

    token_predictor = TokenPredictor(
        data_path=data_path,
        text_input=text_input,
        model_name=model_name,
        reduce_tokens=reduce_tokens,
        first_n_tokens=first_n_tokens
    )
    llm_compressor = LLMCompressor()

    data_tokens = token_predictor.get_data_tokens()
    
    # Initialize sum of probabilities for roaring ranking
    probs_sum = np.zeros(token_predictor.tokenizer.vocab_size, dtype=np.float64)
    
    entropy = 0.0

    prompt_tokens = []
    for i in range(len(data_tokens) - 1):
        prompt_tokens.append(data_tokens[i])
        # pop the first token if the prompt is too long
        if len(prompt_tokens) > context_length:
            prompt_tokens.pop(0)
        
        print(f"\rProcessing token {i+1}/{len(data_tokens)}", end='')

        next_token_actual_index = data_tokens[i+1]

        # Determine which probability distribution to use for the current token
        if roaring_ranking_n is not None and i < roaring_ranking_n:
            # Use full vocabulary for the first N tokens for both probs_sum and compression
            candidate_token_ids_for_compression, probs_values_for_compression = token_predictor.get_full_token_info(prompt_tokens)
            probs_values_for_compression_np = np.array(probs_values_for_compression)

            # Accumulate probabilities for roaring ranking from the full vocabulary
            for idx, prob in zip(candidate_token_ids_for_compression, probs_values_for_compression_np):
                probs_sum[idx] += prob

            # The token index for compressor is the actual token ID
            token_idx_for_compressor = next_token_actual_index
            # The probabilities for compressor are the full probabilities
            probs_for_compressor_np = probs_values_for_compression_np

        else:
            # For all other tokens, or if roaring_ranking_n is not set, use the potentially reduced token set
            candidate_token_ids_for_compression, probs_values_for_compression = token_predictor.get_token_info(prompt_tokens)
            probs_values_for_compression_np = np.array(probs_values_for_compression)

            # Get the index of the next_token_actual_index within the reduced vocabulary
            try:
                token_idx_for_compressor = candidate_token_ids_for_compression.index(next_token_actual_index)
            except ValueError:
                print(f"Warning: next_token_actual_index {next_token_actual_index} not found in candidate_token_ids for token {i+1}. Skipping compression for this token.")
                continue # Skip this token if it's not in the reduced set

            # The probabilities for compressor are the reduced probabilities
            probs_for_compressor_np = probs_values_for_compression_np


        entropy += -(np.log2(probs_values_for_compression_np[token_idx_for_compressor]))

        llm_compressor.next_token(token_idx_for_compressor, probs_for_compressor_np)

    bit_string = llm_compressor.compress()

    final_size = 0
    birmap_size = 0
    rank_list = None

    if reduce_tokens:
        if roaring_ranking_n is not None:
            # Get distinct tokens from the first N tokens based on probs_sum
            # Sort indices in descending order of accumulated probability
            sorted_indices = np.argsort(probs_sum)[::-1]
            
            # Get the top N distinct tokens based on accumulated probability
            # We need to ensure these are actual token IDs, not just indices in sorted_indices
            # The `token_predictor.tokens_list` contains the actual token IDs that were reduced.
            # We need to find the ranks of these `token_predictor.tokens_list` within the `sorted_indices`
            
            # Create a mapping from token ID to its rank in the sorted_indices
            token_id_to_rank = {token_id: rank for rank, token_id in enumerate(sorted_indices)}
            
            # Filter for tokens that are actually in the reduced set and get their ranks
            rank_list = [token_id_to_rank[tid] for tid in token_predictor.tokens_list if tid in token_id_to_rank]
            rank_list.sort() # Roaring bitmap expects sorted list
            
            bitmap_compression = 'roaring_ranking'
            _, birmap_size = token_predictor.get_bitmap(compression=bitmap_compression, ranking=rank_list)
        else:
            bitmap_compression = 'roaring'
            _, birmap_size = token_predictor.get_bitmap(compression=bitmap_compression)
        
        final_size = len(bit_string) + (birmap_size * 8)
    else:
        final_size = len(bit_string)

    decoded_string_length = len(token_predictor.detokenize(data_tokens)) * 8

    print(f"\nTokenized length: {len(data_tokens)} tokens")
    print(f"Length of bit string: {len(bit_string)} bits")
    if reduce_tokens:
        print(f"Bitmap size: {birmap_size * 8} bits")
    print(f"Final compressed size: {final_size} bits")
    print(f"Decoded string length: {decoded_string_length} bits")
    print(f"Compression ratio: {final_size / decoded_string_length * 100:.4f} %")
    print(f"Entropy: {entropy:.4f}")

    return {
        'roaring_ranking_n': roaring_ranking_n,
        'reduce_tokens': reduce_tokens,
        'first_n_tokens': first_n_tokens,
        'context_length': context_length,
        'decoded_length_bits': decoded_string_length,
        'bitmap_size_bits': birmap_size * 8,
        'final_size_bits': final_size,
        'compression_ratio_percent': final_size / decoded_string_length * 100,
        'entropy': entropy
    }

def main():
    parser = argparse.ArgumentParser(description="Run LLM compression with various options.")
    parser.add_argument("--data_path", type=str, required=False,
                        help="The input text file path for LLM inference.")
    parser.add_argument("--text_input", type=str, required=False,
                        help="The direct text input for LLM inference.")
    parser.add_argument("--roaring_ranking_n", type=int, default=None,
                        help="The 'n' value for roaring ranking compression (avg probs of first n tokens).")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="The HuggingFace model name or path.")
    parser.add_argument("--reduce_tokens", type=bool, default=True,
                        help="Whether to restrict the token space to distinct tokens in the input data.")
    parser.add_argument("--first_n_tokens", type=int, default=10000,
                        help="Number of initial tokens to consider if reduce_tokens is True.")
    parser.add_argument("--context_length", type=int, default=1000,
                        help="The maximum context length for the LLM.")

    args = parser.parse_args()
    print(f"starting compression")
    results = run_compression_analysis(
        data_path=args.data_path,
        text_input=args.text_input,
        roaring_ranking_n=args.roaring_ranking_n,
        model_name=args.model_name,
        reduce_tokens=args.reduce_tokens,
        first_n_tokens=args.first_n_tokens,
        context_length=args.context_length
    )
    print("\n--- Results ---")
    for key, value in results.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
