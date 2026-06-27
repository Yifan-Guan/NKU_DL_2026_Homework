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
model_filename = 'dense_model.pth'
results_dir = './results'


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        inter_channels = bn_size * growth_rate

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1,
                               bias=False)
        self.norm2 = nn.BatchNorm2d(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        layers = []
        current_channels = in_channels
        for _ in range(num_layers):
            layer = DenseLayer(current_channels, growth_rate, bn_size, drop_rate)
            layers.append(layer)
            current_channels += growth_rate
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.norm(x)))
        x = self.pool(x)
        return x


class DenseNetNet(nn.Module):
    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16), num_init_features=64,
                 bn_size=4, drop_rate=0.0, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
        )

        num_features = num_init_features
        blocks = []
        for index, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_features, growth_rate, bn_size, drop_rate)
            blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if index != len(block_config) - 1:
                trans = Transition(num_features, num_features // 2)
                blocks.append(trans)
                num_features = num_features // 2

        self.blocks = nn.Sequential(*blocks)
        self.norm_final = nn.BatchNorm2d(num_features)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.features(x)
        x = self.blocks(x)
        x = self.relu_final(self.norm_final(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train_one_epoch(net: DenseNetNet, trainloader: torch.utils.data.DataLoader, epoch: int):
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
    net = DenseNetNet()
    print(net)


def save_model(net: DenseNetNet):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, model_filename)
    torch.save(net.state_dict(), path)


def test_model(net: DenseNetNet, testloader: torch.utils.data.DataLoader, loss_vector: list, accuracy_vector: list):
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

    net = DenseNetNet()
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
    plt.savefig(os.path.join(results_dir, 'densenet_loss_over_epochs.png'))

    plt.figure(figsize=(12, 5))
    plt.plot(range(1, epochs + 1), accuracy_vector, label='Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.savefig(os.path.join(results_dir, 'densenet_accuracy_over_epochs.png'))

    print('Finished Training and Testing')