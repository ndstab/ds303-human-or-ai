import re, pickle
MAX_SEQ_LEN = 450
with open("model/word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # keep only alphabets and spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text):
    return text.split()

def encode_tokens(tokens, word2idx):
    return [word2idx.get(token, word2idx["<UNK>"]) for token in tokens]

def pad_sequence(seq, word2idx, max_len=MAX_SEQ_LEN):
    if len(seq) > max_len:
        return seq[:max_len]
    else:
        return seq + [word2idx["<PAD>"]] * (max_len - len(seq))
