import unicodedata

def get_stats(ids, counts = None):

    #function, which given an array returns a dictiory which counts the number of consecutive pairs
    #[1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}

    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): #iterate over consecutive elements
        counts[pair] = counts.get(pair, 0) + 1

        return counts
    
def merge(ids, pair, indx):
    newids = []
    i = 0
    while i < len(ids):
        #if not at the last position and the pair matches, replace;
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
            newids.append(indx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    
    return newids

#helper functions------------------------------------------------------------------------------------------

def replace_control_characters(s: str) -> str:
    '''
    we would like to avoid printing control characters, as they will fuck with the output.
    https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    http://www.unicode.org/reports/tr44/#GC_Values_Table
    '''
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) #character approved, you can go be appended to chars :)
        else:
            chars.append(f"\\u{ord(ch):04x}") #converts unknown / control characters into their unicode escape sequences

    return "".join(chars)

def render_token(t: bytes) -> str:
    #pretty printing a token, escaping control characters
    s = t.decode("utf-8", errors = "replace")
    s = replace_control_characters(s)

    return s

#------------------tokenizer class---------------------------------------------------------------------------

class Tokenizer:

    def __init__(self):
        #the default is: a vocab size of 256, no merges, and no pairs
        self.merges = {} #(int, int) -> int
        self.pattern = "" # the string
        self.special_tokens = {} #special tokens, example <|endoftext|> : 100257
        self.vocab = self._build_vocab() #int -> bytes

    def train(self, text, vocab_size, verbose = False):
        #tokenizer can train a vocabulary of size_vocab from given text
        raise NotImplementedError

    def encode(self, text):
        #tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids):
        #tokenizer can decode a list of integers into a string
        raise NotImplementedError
    
    def _build_vocab(self):
        #vocab is simply derived from merges
        vocab = {indx: bytes([indx]) for indx in range(256)}
        for (p0, p1), indx in self.merges.items():
            vocab[indx] = vocab[p0] + vocab[p1]

        for special, indx in self.special_tokens.items():
            vocab[indx] = special.encode("utf-8")

        return vocab
    
    def save(self, file_prefix):

        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            #writing the version, pattern and merges;
            f.write('minBPE.v1')
            f.write(f"{self.pattern}\n")
            #writing the special tokens, first their count, and then the tokens
            f.write(f"{len(self.special_tokens)}\n")

            for special, indx in self.special_tokens.items():
                f.write(f"{special} {indx}\n")
            #merges dict
            for indx1, indx2 in self.merges:
                f.write(f"{indx1} {indx2}\n")

        #writing the vocab for people to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {indx: pair for pair, indx in self.merges.items()}
        with open(vocab_file, encoding= "utf-8") as f:
            for indx, token in self.vocab.items():
                #some tokensmay be partial, or flawed or summ, and cannot be decoded with valid strings
                #we replace these tokens with the character �, using the 'errors = "replace"' attributw.
                #this implies that the .vocab file cannot be used in load(), as apparently decoding in this way is v lossy.
                s = render_token(token)
                #find children of 's', if any
                if indx in inverted_merges:
                    #if this token also has children, then we render all of this very cutely as a merge
                    indx0, indx1 = inverted_merges[indx]
                    s0 = render_token(self.vocab(indx0))
                    s1 = render_token(self.vocab[indx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {indx}\n")
                else:
                    #otherwise this is a 'leaf' token, and has no children. so, this should be printed as it is
                    f.write(f"[{s}] {indx}\n")

    def load(self, model_file):
        #inverse function of save, but only for the model file
        assert model_file.endswith(".model")

        merges = {}
        special_tokens = {}
        indx = 256
        with open(model_file, 'r', encoding = "utf-8") as f:
            #reading the version, pattern and the special tokens
            version = f.readline().strip()
            assert version == "minBPE.v1"
            self.pattern = f.readline().strip()
            num_special = int(f.readline().strip())
            
            for _ in range(num_special):
                special, special_indx = f.readline().strip().split()
                special_tokens[special] = int(special_indx)

            #reading the merges
            for line in f:
                indx1, indx2 = map(int, line.split())
                merges[(indx1, indx2)] = indx
                indx += 1
        
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()



