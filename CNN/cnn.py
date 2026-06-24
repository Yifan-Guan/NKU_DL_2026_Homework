import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm

import numpy as np
import matplotlib.pyplot as plt

from data import imshow, classes

device = torch.device("cpu")
model_dir = './model'
model_filename = 'cnn_model.pth'
results_dir = './results'

class CNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 4, 6, 28, 28
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_one_epoch(net: CNNNet,trainloader: torch.utils.data.DataLoader, epoch: int):
    net.to(device)
    net.train()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    running_loss = 0.0

    tqdm_loader = tqdm.tqdm(enumerate(trainloader, 0), total=len(trainloader), desc=f'Epoch {epoch + 1}')
    for i, data in tqdm_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)

        loss = net.criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            tqdm_loader.set_postfix({'loss': running_loss / 2000})
            running_loss = 0.0

def print_model():
    net = CNNNet()
    print(net)

def save_model(net: CNNNet):
    os.makedirs(model_dir, exist_ok=True)
    PATH = os.path.join(model_dir, model_filename)
    torch.save(net.state_dict(), PATH)

def test_model(net: CNNNet, testloader: torch.utils.data.DataLoader, loss_vector: list, accuracy_vector: list):
    net.to(device)
    net.eval()
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    total_loss = 0.0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            total_loss += net.criterion(outputs, labels).item()

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    total_correct = sum(correct_pred.values())
    total_samples = sum(total_pred.values())

    total_loss /= len(testloader)
    total_accuracy = 100 * total_correct / total_samples

    loss_vector.append(total_loss)
    accuracy_vector.append(total_accuracy)

    print(f'Average loss on the test images: {total_loss / len(testloader):.3f}')
    print(f'Accuracy of the network on the 10000 test images: {total_accuracy:.2f} %')

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

def train_and_test_model(trainloader: torch.utils.data.DataLoader, testloader: torch.utils.data.DataLoader,
                         epochs: int = 2):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for training.")
    else:
        device = torch.device("cpu")
        print("Using CPU for training.")

    net = CNNNet()
    net.to(device)

    loss_vector = []
    accuracy_vector = []

    for epoch in range(epochs):  # loop over the dataset multiple times
        train_one_epoch(net, trainloader, epoch)
        save_model(net)
        test_model(net, testloader, loss_vector, accuracy_vector)

    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(range(1, epochs + 1), loss_vector, label='Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.savefig(os.path.join(results_dir, 'cnn_loss_over_epochs.png'))

    plt.figure(figsize=(12, 5))
    plt.plot(range(1, epochs + 1), accuracy_vector, label='Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.savefig(os.path.join(results_dir, 'cnn_accuracy_over_epochs.png'))

    print('Finished Training and Testing')