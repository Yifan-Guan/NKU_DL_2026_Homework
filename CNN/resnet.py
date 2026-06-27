import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm

import matplotlib.pyplot as plt

from data import classes

device = torch.device("cpu")
model_dir = './model'
model_filename = 'resnet_model.pth'
results_dir = './results'


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out


class ResNetNet(nn.Module):
    def __init__(self, block=BasicBlock, layers=(2, 2, 2, 2), num_classes=10):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def _make_layer(self, block, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def train_one_epoch(net: ResNetNet, trainloader: torch.utils.data.DataLoader, epoch: int):
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
    net = ResNetNet()
    print(net)


def save_model(net: ResNetNet):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, model_filename)
    torch.save(net.state_dict(), path)


def test_model(net: ResNetNet, testloader: torch.utils.data.DataLoader, loss_vector: list, accuracy_vector: list):
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

    net = ResNetNet()
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
    plt.savefig(os.path.join(results_dir, 'resnet_loss_over_epochs.png'))

    plt.figure(figsize=(12, 5))
    plt.plot(range(1, epochs + 1), accuracy_vector, label='Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.savefig(os.path.join(results_dir, 'resnet_accuracy_over_epochs.png'))

    print('Finished Training and Testing')