import json
import os
import numpy as np

from arithmetic_coder import ArithmeticDecoder, ArithmeticEncoder, BitInputStream, BitOutputStream


def gen_rank(probs, next_token):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True, stable=True)
    rank_list = []
    if next_token.shape[0] > 1:
        for i in range(next_token.shape[0]):
            rank_list += [torch.where(probs_idx[i:i+1, :] == next_token[i])[-1]]
        rank = torch.squeeze(torch.stack(rank_list))
    else:
        rank = torch.where(probs_idx == next_token)[-1]
    return rank


def read_bitstream(bitin):
    temp_list = []
    while True:
        temp = bitin.read()
        if temp == -1:
            break
        temp_list += [temp]
    temp_arr = np.array(temp_list)
    final_ind = (np.where(temp_arr == 1)[0][-1]).astype(int)
    final_arr = temp_arr[:final_ind+1]

    return final_arr


def print_tokens(tokens):
    decoded_tokens = tokenizer.batch_decode(
        tokens[0:1000],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    print("|".join(decoded_tokens))


def print_probs(probs):
    print("|".join([f"{p:.4f}" for p in probs]))


class LLMzip_encode:
    def __init__(self, model, tokenizer, filename):
        self.model = model
        self.tokenizer = tokenizer

        self.filename = filename
        self.file_out = open(self.filename+'_llmzip_ac.txt', 'wb')
        self.bitout = BitOutputStream(self.file_out)
        self.AC_encoder = ArithmeticEncoder(32, self.bitout)

        self.alphabet_size = self.model.config.vocab_size

        self.token_length = 0
        self.starter_tokens = []

    def encode_batch(self, prompt_tokens):
        print_tokens(prompt_tokens)
        bsz = prompt_tokens.shape[0]

        prompt_size = prompt_tokens.shape[1]

        tokens = torch.full((bsz, prompt_size), self.tokenizer.pad_token_id).long()
        tokens[:bsz, : prompt_size] = torch.tensor(prompt_tokens).long()
        print(tokens)

        cur_pos = prompt_size-1
        prev_pos = 0

        logits = self.model.forward(tokens[:, prev_pos:cur_pos]).logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        rank = gen_rank(probs, next_token=tokens[:, cur_pos])

        probs_np2 = probs.cpu().detach().numpy()
        tokens_np2 = tokens[:, cur_pos].cpu().numpy()
        ranks_np2 = rank.cpu().numpy()

        probs_tok = probs_np2[np.arange(bsz), tokens_np2]
        print_tokens(tokens_np2)
        print_probs(probs_tok)

        cumul = np.zeros(self.model.vocab_size+1, dtype=np.uint64)
        for j in range(bsz):
            prob1 = probs_np2[j]
            cumul[1:] = np.cumsum(prob1*10000000 + 1)
            self.AC_encoder.write(cumul, tokens_np2[j])

        return ranks_np2, probs_tok

    def encode(self, win_size: int):
        if not os.path.exists(self.filename + '_tokens.npy'):
            with open(self.filename, 'r') as f_in:
                text_input = f_in.read()

            tokens_full = np.array(tokenizer.encode(text_input))
            np.save(self.filename + '_tokens.npy', tokens_full)
        else:
            tokens_full = np.load(self.filename + '_tokens.npy')

        tokens_full = tokens_full[0:1000]
        print_tokens(tokens_full)

        win_size_enc = win_size + 1  # additional 1 is to pass the true token apart from the context of win_size
        bsz = 2048

        ranks_list = []
        probs_tok_list = []

        n_runs = tokens_full.size-win_size_enc+1

        tokens_encoded = tokens_full[win_size:win_size+n_runs]
        print(win_size)
        self.starter_tokens = tokens_full[:win_size]
        np.save(self.filename + '_starter_tokens.npy', self.starter_tokens)

        n_batches = np.ceil(n_runs/bsz).astype(int)

        for b_ind in range(n_batches):
            batch_range_start = b_ind*bsz
            batch_range_stop = np.minimum(n_runs, (b_ind+1) * bsz)
            # tokens_batch = np.array([np.concatenate(([tokens_full[0]], tokens_full[i:i+win_size_enc])) for i in range(batch_range_start, batch_range_stop)])
            tokens_batch = np.array([tokens_full[i: i + win_size_enc] for i in range(batch_range_start, batch_range_stop)])
            ranks, probs_tok = self.encode_batch(tokens_batch)
            ranks_list += [ranks]
            probs_tok_list += [probs_tok]

            if (b_ind*bsz*100/n_batches) % 10 == 0:
                print(f'Encoder: Completed {int(b_ind*bsz*100/n_batches)} %')

        ranks_full = np.concatenate(ranks_list, 0).squeeze()
        probs_tok_full = np.concatenate(probs_tok_list, 0).squeeze()

        self.token_length = len(tokens_encoded)

        self.AC_encoder.finish()
        self.bitout.close()
        self.file_out.close()

        self.compute_compression_ratio(tokens_encoded, probs_tok_full)

    def decode(self, win_size: int):
        # Open the compressed bit-stream
        with open(self.filename + '_llmzip_ac.txt', 'rb') as f_in:
            bitin   = BitInputStream(f_in)
            decoder = ArithmeticDecoder(32, bitin)

            # Load initial window
            decoded = list(self.starter_tokens[:win_size])
            n_to_decode = self.token_length

            # Loop through tokens
            for _ in range(n_to_decode):
                # build context from win_size tokens
                ctx = torch.tensor([decoded[-win_size:]])
                with torch.no_grad():
                    logits = self.model(ctx).logits[:, -1, :]
                    probs  = torch.softmax(logits, dim=-1).numpy()[0]

                # rebuild distribution for the next token
                cumul = np.zeros(probs.shape[0] + 1, dtype=np.uint64)
                cumul[1:] = np.cumsum(probs * 10_000_000 + 1)

                # pull one symbol out of the bit-stream
                sym = decoder.read(cumul, alphabet_size=self.alphabet_size)
                decoded.append(int(sym))

        # convert to text
        return self.tokenizer.decode(decoded)


    def compute_compression_ratio(self, tokens_encoded, probs_tok):
        text_encoded = self.tokenizer.decode(tokens_encoded.squeeze().tolist())

        N_T = tokens_encoded.size
        N_C = len(text_encoded)

        df_out = {}
        df_out['characters'] = N_C
        df_out['tokens'] = N_T

        entropy_val = np.sum(-np.log2(probs_tok)) / N_C
        df_out['entropy'] = [f"{entropy_val:.4f}"]

        file_in = open(self.filename+"_llmzip_ac.txt", 'rb')
        bitin = BitInputStream(file_in)
        compressed_bits = read_bitstream(bitin)
        rho_AC = compressed_bits.size/N_C
        print(f'Compression Ratio for Arithmetic Coding :  {rho_AC} bits/char')
        file_in.close()

        df_out['Llama+AC compressed file size'] = compressed_bits.size
        df_out['bits per character'] = rho_AC

        print(df_out)

        with open(self.filename+'_metrics.json', 'w') as file_metrics:
            json.dump(df_out, file_metrics)


Encoder = LLMzip_encode(model, tokenizer, filename='../test.txt')

Encoder.encode(win_size=13)
recovered = Encoder.decode(win_size=13)
print("Recovered text:", recovered)