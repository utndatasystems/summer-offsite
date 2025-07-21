from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
import math
from pyroaring import BitMap

class TokenPredictor:
    def __init__(self, data_path=None, text_input=None, model_name="Qwen/Qwen2.5-0.5B", reduce_tokens=True, chunk_size=None, first_n_tokens=10000):
        """
        Initializes the TokenPredictor class.

        Args:
            data_path (str, optional): Path to the input text file.
            text_input (str, optional): Direct text input.
            model_name (str): HuggingFace model name or path.
            reduce_tokens (bool): Whether to restrict the token space to distinct tokens in the input data.
            chunk_size (int, optional): The size of token chunks for dynamic masking. If None, masking is done on the whole data.
            first_n_tokens (int): Number of initial tokens to consider if reduce_tokens is True.
        """
        if data_path is None and text_input is None:
            raise ValueError("Either data_path or text_input must be provided.")
        if data_path and text_input:
            raise ValueError("Only one of data_path or text_input can be provided.")

        # Load tokenizer and model from cache or download
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=".cache")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=".cache")
        self.model.eval()  # Set model to evaluation mode

        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model.to('cuda')
            print("Model moved to GPU.")
        else:
            print("GPU not available, using CPU.")

        # Load and tokenize input data
        if data_path:
            self.data = self._get_data_from_file(data_path)
        else:
            self.data = text_input
        
        print("Starting tokenization...")
        # Tokenize the entire data, but we might only use a subset based on first_n_tokens
        full_data_tokens = self.tokenizer.encode(self.data, truncation=False) # Do not truncate here
        if first_n_tokens is not None:
             self.data_tokens = full_data_tokens[:first_n_tokens]
        else:
             self.data_tokens = full_data_tokens

        self.reduce_tokens = reduce_tokens
        self.chunk_size = chunk_size
        
        if self.reduce_tokens and self.chunk_size is None:
            # Original behavior: mask based on the first_n_tokens
            self.tokens_list = sorted(list(set(self.data_tokens)))
            self._update_token_mask()
        elif self.chunk_size is not None:
            # Chunking behavior: token_list will be updated dynamically
            self.tokens_list = []
        else:
            # No reduction
            self.tokens_list = list(range(self.tokenizer.vocab_size))

        print(f"Tokenization complete. Total number of tokens: {len(self.data_tokens)}")
        if not self.chunk_size:
            print(f"Number of tokens used for prediction: {len(self.tokens_list)}")

    def set_active_chunk(self, chunk_index):
        """
        Sets the active token mask based on the tokens in the specified chunk.
        Also returns the bitmap and its size for the current chunk.
        """
        if not self.reduce_tokens or self.chunk_size is None:
            print("Warning: set_active_chunk is only effective when reduce_tokens is True and chunk_size is set.")
            return None, 0

        start_index = chunk_index * self.chunk_size
        end_index = min(start_index + self.chunk_size, len(self.data_tokens))
        chunk_tokens = self.data_tokens[start_index:end_index]

        if not chunk_tokens:
            return None, 0

        self.tokens_list = sorted(list(set(chunk_tokens)))
        self._update_token_mask()
        
        print(f"\nActivated chunk {chunk_index}: tokens {start_index}-{end_index}. Distinct tokens: {len(self.tokens_list)}")
        
        # Get the roaring bitmap for the current chunk's token list
        bitmap = BitMap(self.tokens_list)
        binary_data = bitmap.serialize()
        size_bytes = len(binary_data) # pyroaring serialize returns bytes
        
        return binary_data, size_bytes

    def _update_token_mask(self):
        """
        Updates the index tensor and bitmap for the current self.tokens_list.
        """
        self.index_tensor = torch.tensor(
            self.tokens_list, dtype=torch.long, device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        vocab_size = self.tokenizer.vocab_size
        self.token_bitmap = torch.zeros(vocab_size, dtype=torch.bool)
        self.token_bitmap[self.tokens_list] = True


    def _get_data_from_file(self, data_path):
        """
        Load data from a given text file.

        Args:
            data_path (str): File path to the input text.

        Returns:
            str: Contents of the file as a string.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        with open(data_path, 'r') as f:
            return f.read()

    def _get_distinct_tokens(self):
        """
        Get distinct tokens from the input data.

        Returns:
            list[int]: Sorted list of distinct token IDs.
        """
        tokens = self.data_tokens
        return list(set(tokens))

    def get_token_info(self, prompt_tokens):
        """
        Given a list of prompt tokens, return the list of candidate token IDs and their probabilities.

        Args:
            prompt_tokens (list[int]): Tokenized prompt input.

        Returns:
            tuple: (List of candidate token IDs, list of corresponding probabilities)
        """
        input_ids = torch.tensor([prompt_tokens])
        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]  # Get logits for the last token

        logits = logits[0]  # Shape: (vocab_size,)

        if self.reduce_tokens:
            # Select only logits for reduced token set
            selected_logits = torch.index_select(logits, dim=0, index=self.index_tensor)
        else:
            selected_logits = logits

        # Convert logits to probabilities using softmax
        probs = torch.nn.functional.softmax(selected_logits, dim=0).tolist()

        return self.tokens_list, probs

    def get_full_token_info(self, prompt_tokens):
        """
        Given a list of prompt tokens, return the list of all token IDs and their probabilities
        (full vocabulary), regardless of the reduce_tokens setting.

        Args:
            prompt_tokens (list[int]): Tokenized prompt input.

        Returns:
            tuple: (List of all token IDs, list of corresponding probabilities)
        """
        input_ids = torch.tensor([prompt_tokens])
        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]  # Get logits for the last token

        logits = logits[0]  # Shape: (vocab_size,)

        # Convert logits to probabilities using softmax for the full vocabulary
        probs = torch.nn.functional.softmax(logits, dim=0).tolist()

        return list(range(self.tokenizer.vocab_size)), probs

    def get_ids_probs(self, prompt_tokens, next_token):
        """
        Get the index and probability of the next token from the list of predicted tokens.

        Args:
            prompt_tokens (list[int]): Tokenized input prompt.
            next_token (int): Token ID to look for in the prediction.

        Returns:
            tuple: (Index of the next_token in the token list, probability score)
        """
        token_ids, probs = self.get_token_info(prompt_tokens)
        next_index = token_ids.index(next_token)
        return next_index, probs

    def get_data_tokens(self):
        """
        Get the full list of tokens from the loaded data.

        Returns:
            list[int]: List of token IDs from the data.
        """
        return self.data_tokens

    def detokenize(self, token_ids):
        """
        Convert a list of token IDs back to a string.

        Args:
            token_ids (list[int]): List of token IDs.

        Returns:
            str: Decoded string.
        """
        return self.tokenizer.decode(token_ids)

    def get_token_by_id(self, token_id):
        """
        Get the token ID at a given index in the reduced token list.

        Args:
            token_id (int): Index in the reduced token list.

        Returns:
            int: Token ID corresponding to the given index.
        """
        return self.tokens_list[token_id]

    def get_bitmap(self, compression='none', ranking=None):
        """
        Get the bitmap and size with optional compression.

        Args:
            compression (str): Type of compression ('none', 'rle', 'sparse').

        Returns:
            tuple: (bitmap_representation, size_in_bytes)
        """
        if not self.reduce_tokens:
            raise ValueError("Bitmap only available when reduce_tokens is True.")

        if compression == 'none':
            size_bytes = (len(self.token_bitmap) + 7) // 8
            return self.token_bitmap.clone(), size_bytes

        elif compression == 'rle':
            rle = []
            current_bit = self.token_bitmap[0].item()
            count = 1
            count_max = 0
            for bit in self.token_bitmap[1:]:
                bit = bit.item()
                if bit == current_bit:
                    count += 1
                else:
                    if count > count_max:
                        count_max = count
                    rle.append((current_bit, count))
                    current_bit = bit
                    count = 1
            if count > count_max:
                count_max = count
            rle.append((current_bit, count))
            
            size_bits = len(rle) * (1 + math.log2(count_max))  # 1 bit for value, log2 for count
            size_bytes = (size_bits + 7) // 8
            return rle, size_bytes

        elif compression == 'sparse':
            indices = self.tokens_list
            size_bytes = len(indices) * 4  # 4 bytes per int index
            return indices, size_bytes
        
        elif compression == 'roaring':
            bitmap = BitMap(self.tokens_list)
            binary_data = bitmap.serialize()
            # The result of serialize() is a bytes object, so its length is the size in bytes.
            size_bytes = len(binary_data)
            # Verification step
            deserialized_bitmap = BitMap.deserialize(binary_data)
            assert self.tokens_list == list(deserialized_bitmap)
            return binary_data, size_bytes
        
        elif compression == 'roaring_ranking':
            assert ranking is not None
            bitmap = BitMap(ranking)
            binary_data = bitmap.serialize()
            # The result of serialize() is a bytes object, so its length is the size in bytes.
            size_bytes = len(binary_data)
            # Verification step
            deserialized_bitmap = BitMap.deserialize(binary_data)
            assert ranking == list(deserialized_bitmap)
            return binary_data, size_bytes

        else:
            raise ValueError(f"Unsupported compression type: {compression}")
