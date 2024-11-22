import torch
import pickle
import pandas as pd

from syllables_splitting import split
from models import LemmaUsingNet


model = LemmaUsingNet()
max_length = model.get_parameter("MAX_LENGTH")
max_lemma_length = model.get_parameter("MAX_LEMMA_LENGTH")
dict_size = model.get_parameter("DICT_SIZE")
lemma_dict_size = model.get_parameter("LEMMA_DICT_SIZE")
model.load_state_dict(torch.load("checkpoints/bert_net57.pt"))
model.eval()
vocab = pickle.load(open(f"checkpoints/vocab{dict_size}.pth", "rb"))
lemma_vocab = pickle.load(open(f"checkpoints/vocab_lemma{lemma_dict_size}.pth", "rb"))

df = pd.read_csv("data/test.csv")
words = [
    [vocab[j] if j in vocab else vocab["<unk>"] for j in split(i)] for i in df["word"]
]
words = [i + (max_length - len(i)) * [vocab["<pad>"]] for i in words]
words = torch.tensor(words, dtype=torch.long)
lemmas = [
    [lemma_vocab[j] if j in lemma_vocab else lemma_vocab["<unk>"] for j in split(i)]
    for i in df["lemma"]
]
lemmas = [i + (max_lemma_length - len(i)) * [lemma_vocab["<pad>"]] for i in lemmas]
lemmas = torch.tensor(lemmas, dtype=torch.long)
y_pred = model(words, lemmas).argmax(dim=1).tolist()
y_pred = [i + 1 for i in y_pred]
answers = pd.DataFrame({"id": [i for i in range(len(y_pred))], "stress": y_pred})
answers.to_csv("answers.csv", index=False)
