from llm_compressor import LLMCompressor, LLMDecompressor
from prediction import TokenPredictor
import numpy as np


def run_global_mask_experiment(
    data_path,
    model_name="Qwen/Qwen2.5-0.5B",
    context_length=131000,
    first_n_tokens=10000,
    use_kv_cache=False,
    retain_tokens=1000
):
    """
    Runs the compression experiment using a global token mask.

    This function compresses a sequence of tokens using an arithmetic coder guided by
    probabilities from a language model. A global bitmap of possible tokens is used
    to reduce the prediction space.

    Args:
        data_path (str): Path to the data file.
        model_name (str): Name of the language model to use.
        context_length (int): The maximum number of tokens to consider as context.
        first_n_tokens (int): The number of tokens from the dataset to process.
        use_kv_cache (bool): Whether to use the model's KV cache for faster inference.
        retain_tokens (int): Number of tokens to retain when the context length is exceeded (only with KV cache).

    Returns:
        tuple: A tuple containing:
            - str: The compressed bit string.
            - bytes: The serialized bitmap data.
            - dict: A dictionary with compression statistics.
    """
    print(f"\n----- Running Experiment: Global Token Mask (first_n_tokens={first_n_tokens}, kv_cache={use_kv_cache}) -----")
    
    # Initialize the token predictor to get tokens and probabilities from the model.
    token_predictor = TokenPredictor(
        data_path=data_path,
        model_name=model_name,
        reduce_tokens=True,  # Use a reduced vocabulary (global mask)
        chunk_size=None,
        first_n_tokens=None
    )
    
    llm_compressor = LLMCompressor()
    data_tokens = token_predictor.get_data_tokens()[:first_n_tokens]

    # Get the compressed bitmap of the vocabulary and its size.
    bitmask_data, bitmap_size_bytes = token_predictor.get_bitmap(compression='roaring')
    total_bitmap_size = bitmap_size_bytes * 8

    prompt_tokens = []
    # Process each token in the dataset to compress it.
    for i in range(len(data_tokens) - 1):
        prompt_tokens.append(data_tokens[i])

        # Manage the context window to avoid exceeding the model's limit.
        if use_kv_cache:
            # With KV cache, we can keep a shorter, rolling context.
            if len(prompt_tokens) > context_length:
                prompt_tokens = prompt_tokens[-retain_tokens:]
        else:
            # Without KV cache, maintain a sliding window of the full context length.
            if len(prompt_tokens) > context_length:
                prompt_tokens.pop(0)
        
        print(f"\rProcessing token {i+1}/{len(data_tokens)}", end='')
        next_token_actual_index = data_tokens[i+1]
        
        # Get the model's predictions for the next token.
        if use_kv_cache:
            candidate_token_ids, probs_values = token_predictor.get_token_info_cache(
                prompt_tokens,
                use_kv_cache=True
            )
        else:
            candidate_token_ids, probs_values = token_predictor.get_token_info(prompt_tokens)
        
        probs_values_np = np.array(probs_values)

        # Find the index of the actual next token within the candidate tokens.
        try:
            token_idx_for_compressor = candidate_token_ids.index(next_token_actual_index)
        except ValueError:
            print(f"\nFATAL: Token {next_token_actual_index} not in global vocabulary. This should not happen.")
            break
        
        # Provide the actual token's index and the probability distribution to the compressor.
        llm_compressor.next_token(token_idx_for_compressor, probs_values_np)

    # Finalize the compression to get the bit string.
    bit_string = llm_compressor.compress()
    total_arithmetic_code_size = len(bit_string)

    # Calculate final size and compression ratio.
    final_size = total_arithmetic_code_size + total_bitmap_size
    original_size = len(token_predictor.detokenize(data_tokens)) * 8

    return bit_string, bitmask_data, {
        "first_n_tokens": first_n_tokens,
        "chunk_size": -1, # -1 indicates global mask, not chunking
        "arithmetic_code_size_bits": total_arithmetic_code_size,
        "bitmap_size_bits": total_bitmap_size,
        "final_size_bits": final_size,
        "compression_ratio_percent": final_size / original_size * 100,
    }


def run_global_mask_decompression(
    bit_string,
    data_path,
    model_name="Qwen/Qwen2.5-0.5B",
    context_length=131000,
    first_n_tokens=10000,
    use_kv_cache=False,
    retain_tokens=1000
):
    """
    Runs the decompression for a bit string compressed with a global token mask.

    This function reconstructs the original sequence of tokens by using the language
    model to predict probabilities and the arithmetic decompressor to decode the next token.

    Args:
        bit_string (str): The compressed bit string.
        data_path (str): Path to the original data file (for the token predictor).
        model_name (str): Name of the language model to use.
        context_length (int): The maximum number of tokens to consider as context.
        first_n_tokens (int): The number of tokens to decompress.
        use_kv_cache (bool): Whether to use the model's KV cache.
        retain_tokens (int): Number of tokens to retain in the context (with KV cache).

    Returns:
        list: The list of reconstructed token IDs.
    """
    print(f"\n----- Running Decompression: Global Token Mask (first_n_tokens={first_n_tokens}, kv_cache={use_kv_cache}) -----")
    
    # Initialize the token predictor.
    token_predictor = TokenPredictor(
        data_path=data_path,
        model_name=model_name,
        reduce_tokens=True,
        chunk_size=None,
        first_n_tokens=None
    )

    # Get the original tokens to know the starting token and the total length.
    data_tokens = token_predictor.get_data_tokens()[:first_n_tokens]
    
    decompressor = LLMDecompressor(bit_string)
    # Start decompression with the first token from the original data.
    decompress_prompt_tokens = [data_tokens[0]]
    reconstructed_tokens = [data_tokens[0]]

    # Decompress token by token.
    for i in range(1, len(data_tokens)):
        # Maintain the prompt length, following the same policy as compression.
        if use_kv_cache:
            if len(decompress_prompt_tokens) > context_length:
                decompress_prompt_tokens = decompress_prompt_tokens[-retain_tokens:]
        else:
            if len(decompress_prompt_tokens) > context_length:
                decompress_prompt_tokens.pop(0)
        
        # Get the model's probability distribution for the current context.
        if use_kv_cache:
            candidate_token_ids, probs_values = token_predictor.get_token_info_cache(
                decompress_prompt_tokens,
                use_kv_cache=True
            )
        else:
            candidate_token_ids, probs_values = token_predictor.get_token_info(decompress_prompt_tokens)

        probs_values_np = np.array(probs_values)
        
        # Decompress the next token's index from the bit string.
        token_idx = decompressor.decompress(probs_values_np)
        next_token = token_predictor.get_token_by_id(token_idx)
        
        # Append the decompressed token to the context for the next step.
        decompress_prompt_tokens.append(next_token)
        reconstructed_tokens.append(next_token)
        
        print(f"\rDecompressing token {i+1}/{len(data_tokens)}", end='')

    print("\nDecompression completed.")
    
    return reconstructed_tokens

def run_global_mask_compression_decompression_test(
    data_path,
    model_name="Qwen/Qwen2.5-0.5B",
    context_length=131000,
    first_n_tokens=10000,
    use_kv_cache=False,
    retain_tokens=1000
):
    """
    Runs a full compression and decompression cycle and verifies the result.

    This function is a test utility to ensure that the compression is lossless by
    comparing the original tokens with the reconstructed tokens.

    Args:
        data_path (str): Path to the data file.
        model_name (str): Name of the language model.
        context_length (int): Maximum context length.
        first_n_tokens (int): Number of tokens to process.
        use_kv_cache (bool): Whether to use the KV cache.
        retain_tokens (int): Number of tokens to retain in the context.

    Returns:
        dict: A dictionary containing the compression results and a verification status.
    """
    print(f"\n===== Running Compression + Decompression Test (kv_cache={use_kv_cache}) =====")
    
    # Step 1: Compress the data.
    bit_string, _, compression_result = run_global_mask_experiment(
        data_path=data_path,
        model_name=model_name,
        context_length=context_length,
        first_n_tokens=first_n_tokens,
        use_kv_cache=use_kv_cache,
        retain_tokens=retain_tokens
    )
    
    print("\nCompression complete.")
    print(f"Compression Ratio: {compression_result['compression_ratio_percent']:.2f}%")

    # Step 2: Decompress the data.
    reconstructed_tokens = run_global_mask_decompression(
        bit_string=bit_string,
        data_path=data_path,
        model_name=model_name,
        context_length=context_length,
        first_n_tokens=first_n_tokens,
        use_kv_cache=use_kv_cache,
        retain_tokens=retain_tokens
    )
    
    print("\nDecompression complete.")

    # Step 3: Verify that the reconstructed tokens match the original ones.
    token_predictor = TokenPredictor(
        data_path=data_path,
        model_name=model_name,
        reduce_tokens=True,
        chunk_size=None,
        first_n_tokens=None
    )
    original_tokens = token_predictor.get_data_tokens()[:first_n_tokens]
    
    match = reconstructed_tokens == original_tokens
    print(f"Verification: {'✅ PASS' if match else '❌ FAIL'}")
    
    return {
        "compression_result": compression_result,
        "verification_passed": match
    }
