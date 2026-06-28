import os
import sys

from data import trainloader, testloader, classes
from data import show_random_images

# from cnn import print_model, train_and_test_model
# from resnet import print_model, train_and_test_model
# from dense import print_model, train_and_test_model
# from mobilenet import print_model, train_and_test_model
from res2net import print_model, train_and_test_model

if __name__ == '__main__':
    # print_model()
    train_and_test_model(trainloader, testloader, epochs=6)