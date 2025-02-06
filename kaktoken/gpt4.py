"""
Implements the GPT-4 Tokenizer as a light wrapper around the RegexTokenizer.
Note that this is a pretrained tokenizer. By default and inside init(), it
loads the pretrained tokenizer from the `cl100k_base` tokenizer of tiktoken.
"""

import tiktoken
from regex import RegexTokenizer


def bpe(mergeable_ranks, token, max_rank):
    #helper function used in get_gpt4_merges() to reconstruct the merge forest
    parts = [bytes([b]) for b in token]
    while True:
        min_indx = None
        min_rank = None

        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_indx = i
                min_rank = rank

        if min_rank is None and (max_rank is not None and min_rank >= max_rank):
            break
            
        assert min_indx is not None
        parts = parts[:min_indx] + [parts[min_indx] + parts[min_indx + 1]] + parts[min_indx + 2:]
    
    return parts

def recover_merges(mergeable_ranks):
    """
    the 'merges' are already in the byte sequence form, in their merged state.
    we have to recover the original pairings, and we do that by runnnign a BPE training run on all the tokens in order.
    """
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue #skipping the raw bytes
        pair = tuple(bpe(mergeable_ranks, token, max_rank = rank))
        assert len(pair) == 2
        #recovering integer ranks of the pairs
        indx0 = mergeable_ranks(pair[0])
        indx1 = mergeable_ranks(pair[1])
        merges[(indx0, indx1)] = rank

    return merges

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

class GPT4Tokenizer(RegexTokenizer):

    def __init__(self):
        super().__init__(pattern = GPT4_SPLIT_PATTERN)
        #get the official tokenizer ('cl100k_base) and its merges
        enc = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks = enc.mergeable_ranks

        self.merges = recover_merges(mergeable_ranks)
        #reconstructing the vocabulary from the merges
        vocab = {indx: bytes([indx]) for indx in range(256)}
        for (p0, p1), indx in self.merges.items():
            vocab[indx] = vocab[p0] + vocab[p1]
        self.vocab = vocab

        #no idea why, but single byte corresponding tokens are permuted in a different order. we deal with that here
        self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}
        #registering the special tokens (since they are added in the end)
        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    def _encode_chunk(self, text_bytes):
        #before processing the bytes, we have to permute them
        text_bytes = bytes(self.byte_shuffle[b] for b in text_bytes)
        ids = super()._encode_chunk(text_bytes)

        return ids
    
    def decode(self, ids):
        #before decoding we have to 'unpermute' thebytes
        text_bytes = b"".join(self.vocab[indx] for indx in ids)
        text_bytes = bytes(self.inverse_byte_shuffle[b] for b in text_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def train(self, text, vocab_size, verbose = False):
        #the tokenizer cl100k is pretrained, it isn't meant to be trained
        raise NotImplementedError
    
    def save(self, file_prefix):
        raise NotImplementedError("GPT4 Tokenizer cannot be saved.")
    
    def load(self, file_prefix):
        raise NotImplementedError("GPT4 Tokenizer cannot be loaded.")
    

    def save_vocab(self, vocab_file):
        from base import render_token
        #building the vocab keeping in mind the byte shuffle
        vocab = {indx: bytes([self.inverse_byte_shuffle[indx]]) for indx in range(256)}
        for (p0, p1), indx in self.merges.items():
            vocab[indx] = vocab[p0] + vocab[p1]
        #now merge the shuffled bytes and writeeee
        inverted_merges = {indx: pair for pair, indx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for indx, token in vocab.items():
                s = render_token(token)
                if indx in inverted_merges:
                    indx0, indx1 = inverted_merges[indx]
                    s0 = render_token(vocab[indx0])
                    s1 = render_token(vocab[indx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {indx}\n")
                else:
                    f.write(f"[{s}] {indx}\n")


