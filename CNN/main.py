from data import trainloader, testloader, classes
from data import show_random_images
from cnn import print_model, train_and_test_model

if __name__ == '__main__':
    train_and_test_model(trainloader, testloader, epochs=6)