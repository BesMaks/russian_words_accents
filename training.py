import json

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from build_vocab import build_vocab
from models import LemmaUsingNetWithPositionalEmbeddings
from syllables_splitting import split

BATCH_SIZE = 64
EPOCHS = 70
LR = 0.0005
LOG_INTERVAL = 10


class CustomDataset(Dataset):
    def __init__(self, words, labels):
        self.words = list(words["word"])
        self.lemma = list(words["lemma"])
        self.labels = list(labels)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, item):
        word = [
            vocab[i] if i in vocab else vocab["<unk>"] for i in split(self.words[item])
        ]
        lemma = [
            lemma_vocab[i] if i in lemma_vocab else lemma_vocab["<unk>"]
            for i in split(self.lemma[item])
        ]
        if len(word) < max_length:
            word += [vocab["<pad>"]] * (max_length - len(word))
        if len(lemma) < max_lemma_length:
            lemma += [lemma_vocab["<pad>"]] * (max_lemma_length - len(lemma))
        return {
            "word": torch.tensor(word, dtype=torch.long),
            "lemma": torch.tensor(lemma, dtype=torch.long),
            "label": torch.tensor([self.labels[item] - 1]),
        }

df = pd.read_csv("data/cleaned_dataset.csv")
df2 = pd.read_csv("data/test.csv")

# Build vocabularies
vocab = build_vocab(list(df["word"]), list(df2["word"]))
lemma_vocab = build_vocab(list(df["lemma"]), list(df2["lemma"]), tag="lemma")
x_train, x_test, y_train, y_test = train_test_split(
    df[["word", "lemma"]], df["stress"], test_size=0.1, shuffle=True, random_state=42
)

train_dataloader = DataLoader(CustomDataset(x_train, y_train), batch_size=BATCH_SIZE)
test_dataloader = DataLoader(CustomDataset(x_test, y_test), batch_size=BATCH_SIZE)

# Train & evaluate parameters
writer = SummaryWriter()
model = LemmaUsingNetWithPositionalEmbeddings()
max_length = model.get_parameter("MAX_LENGTH")
max_lemma_length = model.get_parameter("MAX_LEMMA_LENGTH")
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
metrics = {
    "test_loss": [],
    "train_loss": [],
    "test_acc": [],
    "train_acc": [],
    "test_precision": [],
    "test_recall": [],
}

# Main loop
for epoch in range(1, EPOCHS + 1):
    model.train()
    mean_loss, mean_acc = 0, 0
    data_count = 0
    
    # Train
    for i, data in enumerate(train_dataloader):
        y_pred = model(data["word"], data["lemma"])
        y_test = data["label"].reshape(len(data["label"]))

        # Loss
        loss = criterion(y_pred, y_test)
        mean_loss += loss.tolist() * len(y_test)
        mean_acc += accuracy_score(
            y_test.tolist(), y_pred.argmax(dim=1).tolist()
        ) * len(y_test)

        data_count += len(y_test)

        # Backward pass and optimization
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    # Metrics to TensorBoard
    train_loss = mean_loss / data_count
    train_acc = mean_acc / data_count
    writer.add_scalar('train_loss', train_loss, epoch)
    writer.add_scalar('train_acc', train_acc, epoch)

    # Save weights
    torch.save(model.state_dict(), f"weights/LemmaUsingNet_{epoch}.pt")

    # Evaluate
    model.eval()
    loss, acc, precision, recall = 0, 0, 0, 0
    data_count = 0
    for i, data in enumerate(test_dataloader):
        y_pred = model(data["word"], data["lemma"])
        y_test = data["label"].reshape(len(data["label"]))

        loss += criterion(y_pred, y_test).tolist() * len(y_pred)

        y_pred = y_pred.argmax(dim=1).tolist()
        y_test = y_test.tolist()

        acc += accuracy_score(y_test, y_pred) * len(y_pred)

        precision += precision_score(
            y_test, y_pred, average="macro", zero_division=True
        ) * len(y_pred)
        recall += recall_score(
            y_test, y_pred, average="macro", zero_division=True
        ) * len(y_pred)

        data_count += len(y_pred)

    # Test metrics to TensorBoard
    test_loss = loss / data_count
    test_acc = acc / data_count
    test_precision = precision / data_count
    test_recall = recall / data_count
    writer.add_scalar('test_loss', test_loss, epoch)
    writer.add_scalar('test_acc', test_acc, epoch)
    writer.add_scalar('test_precision', test_precision, epoch)
    writer.add_scalar('test_recall', test_recall, epoch)

    # Save the metrics to a file
    with open("checkpoints/LemmaUsingNet_metrics.json", "w") as f:
        json.dump(metrics, f)

    print(f"After epoch {epoch}/{EPOCHS}")
    print(f"Acc: {test_acc}, Precision: {test_precision}, Recall: {test_recall}")

writer.close()
