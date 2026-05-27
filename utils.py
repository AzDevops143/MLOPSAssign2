import torch
from sklearn.metrics import accuracy_score, f1_score

# Author: g25ait2026, charantej
# Helpers for MLOps Assignment 2

# Label maps for Goodreads genres
id2label = {
    0: 'poetry',
    1: 'children',
    2: 'comics_graphic',
    3: 'fantasy_paranormal',
    4: 'history_biography',
    5: 'mystery_thriller_crime',
    6: 'romance',
    7: 'young_adult'
}
label2id = {label: id for id, label in id2label.items()}

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }
