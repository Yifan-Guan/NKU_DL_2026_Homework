from data import alldata, train_set, test_set, n_letters
from char_rnn import print_model, train_and_evaluate

if __name__ == "__main__":
    # print_model(n_letters, len(alldata.labels_uniq))
    train_and_evaluate(alldata, n_letters, train_set, test_set)