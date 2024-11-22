import os
import torch
import pickle
import pandas as pd
from tqdm import tqdm

os.chdir("..")
from models import LemmaUsingNet
from syllables_splitting import split

model = LemmaUsingNet()
max_length = model.get_parameter("MAX_LENGTH")
max_lemma_length = model.get_parameter("MAX_LEMMA_LENGTH")
dict_size = model.get_parameter("DICT_SIZE")
lemma_dict_size = model.get_parameter("LEMMA_DICT_SIZE")
model.load_state_dict(torch.load("checkpoints/bert_net40.pt"))
model.eval()
vocab = pickle.load(open(f"checkpoints/vocab{dict_size}.pth", "rb"))
lemma_vocab = pickle.load(open(f"checkpoints/vocab_lemma{lemma_dict_size}.pth", "rb"))

df = pd.read_csv("data/train.csv")
words = [
    [vocab[j] if j in vocab else vocab["<unk>"] for j in split(i)] for i in df["word"]
]
words = [i + (max_length - len(i)) * [vocab["<pad>"]] for i in words]
lemmas = [
    [lemma_vocab[j] if j in lemma_vocab else lemma_vocab["<unk>"] for j in split(i)]
    for i in df["lemma"]
]
lemmas = [i + (max_lemma_length - len(i)) * [lemma_vocab["<pad>"]] for i in lemmas]
predictions = []

for word, lemma in tqdm(zip(words, lemmas)):
    torch_word = torch.tensor([word], dtype=torch.long)
    torch_lemma = torch.tensor([lemma], dtype=torch.long)
    y_pred = model(torch_word, torch_lemma).argmax(dim=1).tolist()[0]
    predictions.append(y_pred + 1)
for i in range(len(predictions)):
    if predictions[i] != df["stress"][i]:
        print(
            f"y_true: {df['stress'][i]}, y_pred: {predictions[i]}, word: {df['word'][i]}, index: {i}"
        )
