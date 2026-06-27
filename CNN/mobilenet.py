import os
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

import matplotlib.pyplot as plt

from data import classes

device = torch.device("cpu")
model_dir = './model'
model_filename = 'mobilenet_model.pth'
results_dir = './results'


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileNetNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            DepthwiseSeparableConv(32, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
            DepthwiseSeparableConv(256, 512, stride=2),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 1024, stride=2),
            DepthwiseSeparableConv(1024, 1024, stride=1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train_one_epoch(net: MobileNetNet, trainloader: torch.utils.data.DataLoader, epoch: int):
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
        if i % 2000 == 1999:
            tqdm_loader.set_postfix({'loss': running_loss / 2000})
            running_loss = 0.0


def print_model():
    net = MobileNetNet()
    print(net)


def save_model(net: MobileNetNet):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, model_filename)
    torch.save(net.state_dict(), path)


def test_model(net: MobileNetNet, testloader: torch.utils.data.DataLoader, loss_vector: list, accuracy_vector: list):
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

    print(f'Average loss on the test images: {total_loss:.3f}')
    print(f'Accuracy of the network on the 10000 test images: {total_accuracy:.2f} %')

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


def train_and_test_model(trainloader: torch.utils.data.DataLoader, testloader: torch.utils.data.DataLoader,
                         epochs: int = 2):
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for training.")
    else:
        device = torch.device("cpu")
        print("Using CPU for training.")

    net = MobileNetNet()
    net.to(device)

    loss_vector = []
    accuracy_vector = []

    for epoch in range(epochs):
        train_one_epoch(net, trainloader, epoch)
        save_model(net)
        test_model(net, testloader, loss_vector, accuracy_vector)

    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(range(1, epochs + 1), loss_vector, label='Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.savefig(os.path.join(results_dir, 'mobilenet_loss_over_epochs.png'))

    plt.figure(figsize=(12, 5))
    plt.plot(range(1, epochs + 1), accuracy_vector, label='Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.savefig(os.path.join(results_dir, 'mobilenet_accuracy_over_epochs.png'))

    print('Finished Training and Testing')