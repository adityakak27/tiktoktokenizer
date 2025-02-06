from base import Tokenizer, get_stats, merge

#this is vvvvv minimal byte level BPE tokenizer
#does not handle regex splitting pattern, nor does this handle any special tokens

class BasicTokenizer:
    def __init__(self):
        super().__init__()
    
    def train(self, text, vocab_size, verbose = False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        #preprocessin' the input text
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        #iteratively merge most common pairs to replace them with new tokens
        merges = {} #(int, int) -> int
        vocab = {indx: bytes([indx]) for indx in range(256)} # int -> bytes
        
        for i in range(num_merges):
            #count the number of times every consecutive pair appears
            stats = get_stats(ids)
            #find pair with highest count
            pair = max(stats, key=stats.get)
            #create new token, assign it to the next available id
            indx = 256 + i
            #replace all the occurring instances of the pair in ids with indx
            ids = merge(ids, pair, indx)
            #saving the merge
            merges[pair] = indx
            vocab[indx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {indx} ({vocab[indx]}) had {stats[pair]} occurrences")

        #saving the class variables
        self.merges = merges #used in encode()
        self.vocab = vocab   #used in decode()

    def decode(self, ids):
        #given ids, return a string (ids a list of integers)

        text_bytes = b"".join(self.vocab[indx] for indx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    def encode(self, text):
        #given a string, return token(s)
        text_bytes = text.encode("utf-8") #raaaaw bytes
        ids = list(text_bytes) # list of integers in range 0..255
        while len(ids) >= 2:
            #find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key = lambda p: self.merges.get(p, float("inf")))
            #subtle: if there are no more merges available, the key will
            #result in an inf for every single pair, and the min will be
            #just the first pair in the list, arbitrarily
            #we can detect this terminating case by a membership check
            if pair not in self.merges:
                break #nothing else can be merged anymore
            #otherwise let's merge the best pair (lowest merge index)
            indx = self.merges[pair]
            ids = merge(ids, pair, indx)
        return ids

       