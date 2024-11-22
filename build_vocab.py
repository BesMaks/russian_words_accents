import pickle
from typing import List
import torchtext.vocab
from torchtext.vocab import build_vocab_from_iterator
from syllables_splitting import split


def build_vocab(
    train_words: List[str], test_words: List[str], tag: str = ""
) -> torchtext.vocab.Vocab:
    def tokens_iterator(words):
        for i in words:
            yield [j if wc[j] > 1 or j in test_wc.keys() else "<unk>" for j in split(i)]

    wc, test_wc = {}, {}
    for j in train_words:
        for i in split(j):
            if i in wc.keys():
                wc[i] += 1
            else:
                wc[i] = 1
    for j in test_words:
        for i in split(j):
            if i in test_wc.keys():
                test_wc[i] += 1
            else:
                test_wc[i] = 1
    vocab = build_vocab_from_iterator(
        tokens_iterator(train_words), specials=["<unk>", "<pad>"]
    )
    pickle.dump(vocab, open(f"checkpoints/vocab_{tag}{len(vocab)}.pth", "wb"))
    return vocab
