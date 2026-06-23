import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using PyTorch version: {torch.__version__}, Device: {device}")

batch_size = 32

train_dataset = datasets.MNIST(
    root = './data',
    train = True,
    download = True,
    transform = transforms.ToTensor()
)

validation_dataset = datasets.MNIST(
    root = './data',
    train = False,
    transform = transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True
)

validation_loader = torch.utils.data.DataLoader(
    dataset = validation_dataset,
    batch_size = batch_size,
    shuffle = False
)

def show_data_info():
    for (X_train, y_train) in train_loader:
        print(f"X train: {X_train.size()}, type: {X_train.type()}")
        print(f"y train: {y_train.size()}, type: {y_train.type()}")
        break

    pltsize = 1
    plt.figure(figsize=(10 * pltsize, pltsize))

    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.axis('off')
        plt.imshow(X_train[i].numpy().squeeze(), cmap='gray_r')
        plt.title(f"Class: {y_train[i].item()}")

    plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc1_dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, 80)
        self.fc2_dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(80, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc1_dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_dropout(x)
        x = self.fc3(x)
        return x

model = Net().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()

def train(epoch, log_interval=200):
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        
        output = model(data)

        loss = criterion(output, target)
        
        loss.backward()
        
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}",
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

def validate(loss_vector, accuracy_vector):
    model.eval()
    val_loss = 0
    correct = 0
    
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
    
    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    print(f"\nValidation set: Average loss: {val_loss:.4f},",
          f"Accuracy: {correct}/{len(validation_loader.dataset)}")

    
if __name__ == "__main__":
    print(model)

    epochs = 3
    
    lossv = []
    accv = []

    for epoch in range(1, epochs + 1):
        train(epoch)
        validate(lossv, accv)

    plt.figure(figsize=(5,3))
    plt.plot(np.arange(1, epochs+1), lossv)
    plt.title('validation loss')

    plt.figure(figsize=(5,3))
    plt.plot(np.arange(1, epochs+1), accv)
    plt.title('validation accuracy');

    plt.show()