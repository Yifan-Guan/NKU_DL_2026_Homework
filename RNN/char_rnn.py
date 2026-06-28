import os
import time
import tqdm
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

model_dir = "models"
model_filename = "char_rnn_model.pt"
result_dir = "results"

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, line_tensor):
        rnn_out, hidden = self.rnn(line_tensor)
        output = self.h2o(hidden[0])
        output = self.softmax(output)

        return output

class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, line_tensor):
        lstm_out, (hidden, cell) = self.lstm(line_tensor)
        output = self.h2o(hidden[0])
        output = self.softmax(output)

        return output

def print_model(input_size, output_size, hidden_size=128, model_type="lstm"):
    if model_type.lower() == "rnn":
        model = CharRNN(input_size, hidden_size, output_size)
    else:
        model = CharLSTM(input_size, hidden_size, output_size)
    print(model)

def label_from_output(output, output_labels):
    top_n, top_i = output.topk(1)
    label_i = top_i[0].item()
    return output_labels[label_i], label_i

def train(rnn, device, all_data, training_data, n_epoch = 10, n_batch_size = 64, report_every = 50, learning_rate = 0.2, criterion = nn.NLLLoss(), model_name="Char_RNN"):
    """
    Learn on a batch of training_data for a specified number of iterations and reporting thresholds
    """
    # Keep track of losses for plotting
    current_loss = 0
    current_accuracy = 0
    all_losses = []
    all_accuracies = []
    rnn.train()
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

    start = time.time()
    print(f"training on data set with n = {len(training_data)}")

    for iter in range(1, n_epoch + 1):
        rnn.zero_grad() # clear the gradients

        # create some minibatches
        # we cannot use dataloaders because each of our names is a different length
        batches = list(range(len(training_data)))
        random.shuffle(batches)
        batches = np.array_split(batches, len(batches) // n_batch_size )

        pbar = tqdm.tqdm(enumerate(batches), total=len(batches), desc=f"Epoch {iter}/{n_epoch}")
        for idx, batch in pbar:
            batch_loss = 0
            batch_accuracy = 0
            for i in batch: #for each example in this batch
                (label_tensor, text_tensor, label, text) = training_data[i]
                label_tensor = label_tensor.to(device)
                text_tensor = text_tensor.to(device)

                output = rnn.forward(text_tensor)
                guess, guess_i = label_from_output(output, all_data.labels_uniq)
                if guess == label:
                    batch_accuracy += 1

                loss = criterion(output, label_tensor)
                batch_loss += loss

            # optimize parameters
            batch_loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

            current_loss += batch_loss.item() / len(batch)
            current_accuracy += batch_accuracy / len(batch)

        all_losses.append(current_loss / len(batches) )
        all_accuracies.append(current_accuracy / len(batches))
        current_loss = 0
        current_accuracy = 0

    plt.figure()
    plt.plot(all_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss over time")
    plt.savefig(os.path.join(result_dir, f"{model_name}_loss_over_time.png"))

    plt.figure()
    plt.plot(all_accuracies)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Accuracy over time")
    plt.savefig(os.path.join(result_dir, f"{model_name}_accuracy_over_time.png"))

def evaluate(rnn, device, testing_data, classes, model_name="Char_RNN"):
    confusion = torch.zeros(len(classes), len(classes))

    rnn.eval() #set to eval mode
    with torch.no_grad(): # do not record the gradients during eval phase
        for i in range(len(testing_data)):
            (label_tensor, text_tensor, label, text) = testing_data[i]
            label_tensor = label_tensor.to(device)
            text_tensor = text_tensor.to(device)
            output = rnn(text_tensor)
            guess, guess_i = label_from_output(output, classes)
            label_i = classes.index(label)
            confusion[label_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(len(classes)):
        denom = confusion[i].sum()
        if denom > 0:
            confusion[i] = confusion[i] / denom

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.cpu().numpy()) #numpy uses cpu here so we need to use a cpu version
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=90)
    ax.set_yticks(np.arange(len(classes)), labels=classes)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(os.path.join(result_dir, f"{model_name}_confusion_matrix.png"))


def train_and_evaluate(all_data, n_letters, train_set, test_set, n_hidden=128, n_epoch=27, learning_rate=0.15, report_every=5, model_type="lstm"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if model_type.lower() == "rnn":
        rnn = CharRNN(n_letters, n_hidden, len(all_data.labels_uniq)).to(device)
        model_name = "Char_RNN"
    else:
        rnn = CharLSTM(n_letters, n_hidden, len(all_data.labels_uniq)).to(device)
        model_name = "Char_LSTM"

    train(rnn, device, all_data, train_set, n_epoch=n_epoch, learning_rate=learning_rate, report_every=report_every, model_name=model_name)
    evaluate(rnn, device, test_set, classes=all_data.labels_uniq, model_name=model_name)