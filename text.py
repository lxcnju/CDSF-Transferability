import re
import nltk
from collections import Counter
import numpy as np
import torchtext

try:
    import moxing as mox
    open = mox.file.File
except Exception:
    pass


def tokenize(sent):
    tokens = nltk.word_tokenize(sent)
    for index, token in enumerate(tokens):
        if token == '@' and (index + 1) < len(tokens):
            tokens[index + 1] = '@' + re.sub('[0-9]+.*', '', tokens[index + 1])
            tokens.pop(index)
    return tokens


def get_vocab(corpus, n_vocab=5000, basic_vocab=["<unk>", "<pad>"]):
    words = []
    for wlist in corpus:
        words.extend(wlist)

    wcnt = Counter(words)
    print("Unique words: {}".format(len(wcnt)))

    pairs = wcnt.most_common()
    basic_set = set(basic_vocab)
    corpus_vocab = [word for (word, _) in pairs if word not in basic_set]

    assert n_vocab >= len(basic_vocab)
    corpus_vocab = corpus_vocab[0:n_vocab - len(basic_vocab)]

    vocab = basic_vocab + corpus_vocab
    vocab2int = {word: i for i, word in enumerate(vocab)}
    print("Len of basic vocab: {}, vocab: {}".format(
        len(basic_vocab), len(vocab2int)
    ))
    print(vocab[0:100])
    return vocab2int


def tokens2indices(tokens, vocab2int):
    indices = []
    for word in tokens:
        if word in vocab2int:
            indices.append(vocab2int[word])
        else:
            indices.append(vocab2int['<unk>'])

    return indices


def normalize_length(tokens, vocab2int, max_len=64):
    n_len = len(tokens)
    if n_len >= max_len:
        tokens = tokens[0:max_len]
    else:
        pads = [vocab2int["<pad>"]] * (max_len - n_len)
        tokens.extend(pads)
    return tokens


class PretrainWV():
    """ name: ["glove840b_300", "glove6b_50", "glove6b_200", "glove6b_300",
                "wikien_300", "ngram_100"]
    """
    SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[CLS]", "[B]", "[I]", "[O]"]

    def __init__(self, wv_dir, name):
        self.wv_dir = wv_dir
        self.name = name
        self.dim = int(name.split("_")[1])
        self.wv_model = self.load_pretrain_wv(wv_dir, name)
        self.stoi = self.wv_model.stoi

    def get_vecs_for_tokens(self, tokens):
        vecs = np.array([
            self.get_vec_single_token(tk) for tk in tokens
        ])
        return vecs

    def get_cnts_in_pretrain(self, tokens):
        total = len(tokens)
        if "glove" in self.name or "wikien" in self.name:
            cnt = 0
            for tk in tokens:
                if tk in self.stoi or tk.lower() in self.stoi:
                    cnt += 1
        elif "ngram" in self.name:
            cnt = total
        else:
            raise ValueError("No such wv name.")
        return total, cnt

    def get_vec_single_token(self, token):
        if token in self.SPECIAL_TOKENS:
            vec = np.random.randn(self.dim)
        else:
            if "glove" in self.name or "wikien" in self.name:
                if token in self.stoi:
                    vec = self.wv_model.get_vecs_by_tokens(token)
                    vec = vec.numpy().reshape(-1)
                elif token.lower() in self.stoi:
                    vec = self.wv_model.get_vecs_by_tokens(token.lower())
                    vec = vec.numpy().reshape(-1)
                else:
                    vec = np.random.randn(self.dim)
            elif "ngram" in self.name:
                vec = self.wv_model[token]
                vec = vec.numpy().reshape(-1)
            else:
                raise ValueError("No such wv name.")
        return vec

    def load_pretrain_wv(self, wv_dir, name):
        if name == "glove6b_50":
            wv_model = torchtext.vocab.GloVe(
                name="6B", dim=50, cache=wv_dir
            )
        elif name == "glove6b_200":
            wv_model = torchtext.vocab.GloVe(
                name="6B", dim=200, cache=wv_dir
            )
        elif name == "glove6b_300":
            wv_model = torchtext.vocab.GloVe(
                name="6B", dim=300, cache=wv_dir
            )
        elif name == "glove840b_300":
            wv_model = torchtext.vocab.GloVe(
                name="840B", dim=300, cache=wv_dir
            )
        elif name == "wikien_300":
            wv_model = torchtext.vocab.FastText(
                language="en", cache=wv_dir
            )
        elif name == "ngram_100":
            wv_model = torchtext.vocab.CharNGram(
                cache=wv_dir
            )
        else:
            raise ValueError("No such wv name: {}".format(name))
        return wv_model
