from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch

class TokenPredictor:
    def __init__(self, data_path, model_name="Qwen/Qwen2.5-0.5B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=".cache")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=".cache")

        self.data = self._get_data(data_path)
        self.distinct_tokens = self._get_distinct_tokens()


    def _get_data(self, data_path):
        with open(data_path, 'r', encoding='ascii') as f:
            data = f.read()
        return data

    def _get_distinct_tokens(self):

        print("Starting tokenization...")
        self.data_tokens = self.tokenizer.encode(self.data)
        print(f"Tokenization complete. Total number of tokens: {len(self.data_tokens)}")
        distinct_tokens = list(set(self.data_tokens))
        print(f"Number of distinct tokens: {len(distinct_tokens)}")
        return distinct_tokens

    def get_token_info(self, prompt_tokens):
        input_ids = torch.tensor([prompt_tokens])
        with torch.no_grad():
            outputs = self.model(input_ids)
        
        logits = outputs.logits[:, -1, :]  # shape: (1, vocab_size)
        logits = logits[0]  # shape: (vocab_size,)
        
        # Get logits only for distinct tokens
        filtered_logits = {token_id: logits[token_id].item() for token_id in self.distinct_tokens}
        
        # Get top-k by logit value
        topk_items = sorted(filtered_logits.items(), key=lambda x: x[0], reverse=True)
        
        # Decode token IDs to strings
        top_tokens = [self.tokenizer.decode([tid]) for tid, _ in topk_items]
        top_ids = [tid for tid, _ in topk_items]
        # softmax values (logits)
        # are the second element in the tuple
        # Convert logits to softmax values
        softmax_values = torch.nn.functional.softmax(torch.tensor([val for _, val in topk_items]), dim=0).tolist()
        
        return top_tokens, top_ids, softmax_values

    def get_ids_probs(self, prompt_tokens, next_token):
        _, top_ids, top_values = self.get_token_info(prompt_tokens)
        # locate where is the next token in the top ids
        next_token_index = top_ids.index(next_token)
        return next_token_index, top_values

    def get_data_tokens(self):
        return self.data_tokens

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def get_token_by_id(self, token_id):
        return self.distinct_tokens[token_id]