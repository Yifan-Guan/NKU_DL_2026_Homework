import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from data import dataloader, dataset
from gan import train, print_model

if __name__ == "__main__":
    print_model()
    epochs = 6
    train(dataloader, epochs)