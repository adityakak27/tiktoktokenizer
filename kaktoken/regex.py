import regex as re
from base import Tokenizer, get_stats, merge

# he main GPT text split patterns
#https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):

    def __init__(self, pattern = None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose = False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        #splitting text into chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        #input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        #merging most common pairs to create newer tokens
        merges = {} #(int, int) -> int
        vocab = {indx: bytes([indx]) for indx in range(256)} # indx -> bytes

        for i in range (num_merges):
            #count the number of times consecutive pairs appear
            stats = {}
            for chunk_ids in ids:
                #update in place; by passing it into stats
                get_stats(chunk_ids, stats)
            
            #find the pair that occurrs most frequently
            pair = max(stats, key = stats.get)
            #creating a new token for above mentioned pair
            indx = 256 + i
            #REPLACE!!!!
            ids = [merge(chunk_ids, pair, indx) for chunk_ids in ids]
            #save the merhe
            merges[pair] = indx
            vocab[indx] = vocab[pair[0]] + vocab[pair[1]]
            
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {indx} ({vocab[indx]}) had {stats[pair]} occurrences")

        #class variables
        self.merges = merges 
        self.vocab = vocab   

    
    def register_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        part_bytes = []
        for indx in ids:
            if indx in self.vocab:
                part_bytes.append(self.vocab[indx])
            elif indx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[indx].encode("utf-8"))
            else:
                raise ValueError(f"Invalid token ID : {indx}")
        
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        while len(ids) >= 2:
            #find pair w lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            """
            if there are no more merges available, everyhthing except the first pair will result in an inf
            and the minimum will be the first pair/ we detect this temrinating case by a membership check
            """
            if pair not in self.merges:
                break #nothing else to merge anymore
            indx = self.merges[pair]
            ids = merge(ids, pair, indx)
        return ids
    
    def encode_normal(self, text):
        #encoding that ignores special tokens
        text_chunks = re.findall(self.compiled_pattern, text)
        #chunks of text encoded separately, then joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") #raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids
    
    def encode(self, text, allowed_special = "none_raise"):
        """
        unlike the above function, this handles special characters as well. 
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text.
        iirc, this is also the current tiktoken behaviour. any other behaviour is beyond my scope / annoying
        """
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            #if no special tokens, just use the normal encoding
            return self.encode_normal(text)

        #otherwise, we have to be careful with potential special tokens in text
        #we handle special tokens by splitting the text
        #based on the occurrence of any exact match with any of the special tokens
        #we can use re.split for this. note that surrounding the pattern with ()
        #makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        #now all the special characters are separated from the rest of the text
        #all chunks of text are encoded separately, then joined
        ids = []
        for part in special_chunks:
            if part in special:
                #a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                #an normal sequence, encode it normally
                ids.extend(self.encode_normal(part))
        return ids

