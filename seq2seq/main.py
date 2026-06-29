import os
import torch
from data import get_dataloader
from seq2seq import EncoderRNN, DecoderRNN, AttnDecoderRNN, train, evaluateRandomly, evaluateAndShowAttention
from utils import showPlot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results_dir = "results"

if __name__ == "__main__":
    model_type = "attention"

    hidden_size = 64
    batch_size = 32
    n_epochs = 20

    input_lang, output_lang, train_dataloader = get_dataloader(batch_size)

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    if model_type == "RNN":
        decoder = DecoderRNN(hidden_size, output_lang.n_words).to(device)
    else:
        decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

    print("=" * 20)
    print("Encoder structure:")
    print(encoder)
    print(f"{model_type} Decoder structure:")
    print(decoder)
    print("=" * 20)
    print()

    plot_losses = train(train_dataloader, encoder, decoder, n_epochs, print_every=1, plot_every=1)
    showPlot(plot_losses, os.path.join(results_dir, f'{model_type}_loss_plot.png'), f'{model_type} Loss')

    evaluateRandomly(encoder, decoder, input_lang, output_lang, n=10, output_file=os.path.join(results_dir, f'{model_type}_random_evaluation.txt'))

    if model_type == "attention":
        attention_evaluate_sentences = [
            'il n est pas aussi grand que son pere',
            'je suis trop fatigue pour conduire',
            'je suis desole si c est une question idiote',
            'je suis reellement fiere de vous'
        ]

        for idx, sentence in enumerate(attention_evaluate_sentences):
            print(f"Evaluating sentence {idx + 1}: {sentence}")
            evaluateAndShowAttention(sentence, encoder, decoder, input_lang, output_lang,
                                     os.path.join(results_dir, f'{model_type}_attention_{idx + 1}.png'),
                                     f'{model_type} Attention for "{sentence}"')