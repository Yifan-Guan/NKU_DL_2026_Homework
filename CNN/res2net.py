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
model_filename = 'res2net_model.pth'
results_dir = './results'


class Res2NetBottle2neck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, scale=4, base_width=26, stype='normal'):
        super().__init__()
        if scale < 2:
            raise ValueError('scale must be >= 2')

        self.scale = scale
        self.width = max(int(out_channels * (base_width / 64.0)), 1)
        self.stride = stride
        self.stype = stype
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels, self.width * scale, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(self.width * scale)

        if scale != 1:
            self.nums = scale - 1
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            for i in range(self.nums):
                self.convs.append(nn.Conv2d(self.width, self.width, kernel_size=3, stride=1,
                                            padding=1, bias=False))
                self.bns.append(nn.BatchNorm2d(self.width))

        self.conv3 = nn.Conv2d(self.width * scale, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        if self.downsample is None and (stride != 1 or in_channels != out_channels * self.expansion):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))

        spx = torch.chunk(out, self.scale, dim=1)
        outputs = []
        for i in range(self.scale):
            if i == 0:
                outputs.append(spx[i])
            else:
                current = spx[i]
                current = current + outputs[i - 1]
                current = self.relu(self.bns[i - 1](self.convs[i - 1](current)))
                outputs.append(current)

        out = torch.cat(outputs, dim=1)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Res2NetNet(nn.Module):
    def __init__(self, layers=(2, 2, 2, 2), num_classes=10, scale=4, base_width=26):
        super().__init__()
        self.in_channels = 64
        self.scale = scale
        self.base_width = base_width

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, layers[0], stride=1, stype='stage')
        self.layer2 = self._make_layer(128, layers[1], stride=2, stype='stage')
        self.layer3 = self._make_layer(256, layers[2], stride=2, stype='stage')
        self.layer4 = self._make_layer(512, layers[3], stride=2, stype='stage')
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Res2NetBottle2neck.expansion, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def _make_layer(self, out_channels, blocks, stride, stype):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * Res2NetBottle2neck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * Res2NetBottle2neck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Res2NetBottle2neck.expansion),
            )

        layers = [Res2NetBottle2neck(self.in_channels, out_channels, stride=stride,
                                     downsample=downsample, scale=self.scale, base_width=self.base_width,
                                     stype=stype)]
        self.in_channels = out_channels * Res2NetBottle2neck.expansion
        for _ in range(1, blocks):
            layers.append(Res2NetBottle2neck(self.in_channels, out_channels,
                                             scale=self.scale, base_width=self.base_width, stype='normal'))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def train_one_epoch(net: Res2NetNet, trainloader: torch.utils.data.DataLoader, epoch: int):
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
    net = Res2NetNet()
    print(net)


def save_model(net: Res2NetNet):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, model_filename)
    torch.save(net.state_dict(), path)


def test_model(net: Res2NetNet, testloader: torch.utils.data.DataLoader, loss_vector: list, accuracy_vector: list):
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

    net = Res2NetNet()
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
    plt.savefig(os.path.join(results_dir, 'res2net_loss_over_epochs.png'))

    plt.figure(figsize=(12, 5))
    plt.plot(range(1, epochs + 1), accuracy_vector, label='Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.savefig(os.path.join(results_dir, 'res2net_accuracy_over_epochs.png'))

    print('Finished Training and Testing')