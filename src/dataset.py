import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler


class QADataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.data_len = len(data)

        self.tokenizer = tokenizer
        self.yes_idx = tokenizer("Yes").input_ids[0]
        self.no_idx = tokenizer("No").input_ids[0]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        qa, labels = self.data.iloc[idx]

        return qa, labels


# Pads all examples in batch to same dimension
class PadCollate():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.yes_idx = tokenizer("Yes").input_ids[0]
        self.no_idx = tokenizer("No").input_ids[0]

    def __call__(self, batch):
        qa, labels = zip(*batch)

        # Pad input
        x = self.tokenizer(qa, padding=True, return_tensors="pt")
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]

        return input_ids, attention_mask, torch.tensor(labels)


def create_qa_dataloaders(input_filepath, tokenizer, train_prop, batch_size, shuffle):
    """
    Returns two PyTorch Dataloaders for the dataset: 
    one for training and one for testing. 
    """
    data = pd.read_csv(input_filepath)
    dataset = QADataset(data, tokenizer)

    # Create splits
    indices = list(range(len(dataset)))
    if shuffle:
        np.random.shuffle(indices)

    train_split = int(np.floor(train_prop * len(dataset)))
    train_indices, test_indices = indices[:train_split], indices[train_split:]

    train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=PadCollate(tokenizer), sampler=SubsetRandomSampler(train_indices))
    test_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=PadCollate(tokenizer), sampler=SequentialSampler(test_indices))

    return train_loader, test_loader
