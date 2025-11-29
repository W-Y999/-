import torch
from torch.utils.data import DataLoader
from model import MultiModalRegressor
from dataset import QADataDataset

def train():
    # TODO: config hyperparameters
    # TODO: prepare tokenizer
    # TODO: create dataset & dataloader

    # TODO: initialize model
    model = MultiModalRegressor()

    # TODO: define optimizer & loss
    # optimizer = ...
    # criterion = ...

    # TODO: training loop
    for epoch in range(10):
        for batch in DataLoader(...):
            # TODO: unpack batch
            # TODO: forward
            # TODO: compute loss
            # TODO: backward
            # TODO: optimizer.step()
            pass

        print(f"Epoch {epoch} finished.")

    # TODO: save model checkpoint
    pass


if __name__ == "__main__":
    train()
