"""
Centralized training with Next Word Prediction using LSTM (PyTorch)
Referencing: https://www.kaggle.com/code/dota2player/next-word-prediction-with-lstm-pytorch
"""

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import demoji
import random
import matplotlib.pyplot as plt

import torch
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.nn.functional import one_hot
import torch.optim as optim


# Initialize 
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# N-Grams Generation
def ngram_generate(tokenized_titles, features_vocab, target_vocab):
    def text_to_numerical_sequence(tokenized_text):
        tokens_list = []
        if tokenized_text[-1] in target_vocab.get_itos(): # Check if last token is in target vocab
            for token in tokenized_text[:-1]:
                num_token = features_vocab[token] if token in features_vocab.get_itos() else features_vocab['<oov>']
                tokens_list.append(num_token)
            num_token = target_vocab[tokenized_text[-1]]
            tokens_list.append(num_token)
            return tokens_list
        return None
    
    def add_random_oov_tokens(ngram):
        for idx, _ in enumerate(ngram[:-1]):
            if random.uniform(0, 1) < 0.1:
                ngram[idx] = '<oov>'
        return ngram
    
    def make_ngrams(tokenized_title):
        list_ngrams = []
        for i in range(1, len(tokenized_title)):
            ngram_sequence = tokenized_title[:i+1]
            list_ngrams.append(ngram_sequence)
        return list_ngrams

    # Make N-Grams
    ngrams_list = []
    for tokenized_title in tokenized_titles:
        ngrams_list.extend(make_ngrams(tokenized_title))
    
    # Simulate and Handle OOV Tokens
    ngrams_list_oov = []
    for ngram in ngrams_list:
        ngrams_list_oov.append(add_random_oov_tokens(ngram))
    
    # Convert Text to Numerical Sequences
    return [text_to_numerical_sequence(sequence) for sequence in ngrams_list_oov if text_to_numerical_sequence(sequence)]


# Data Preprocessing
def load_data():
    # Load Data
    df = pd.read_csv('medium_data.csv')
    df_titles = df['title']

    def preprocess(title):
        title = BeautifulSoup(title, 'html.parser').get_text() # Remove HTML tags
        demoji.replace(title, '') # Remove Emojis
        title = re.sub(r"[^a-zA-Z]", " ", title.lower()) # Lowercase and remove non-alphabetic characters
        title.replace(u'\xa0', u' ') # Remove non-breaking space
        title.replace('\x200a', ' ') # Remove zero-width space
        return title
    
    df_titles = df_titles.apply(preprocess)

    # Tokenization
    tokenizer = get_tokenizer('basic_english')
    tokenized_titles = [tokenizer(title) for title in df_titles]

    # Build Vocabulary
    features_vocab = build_vocab_from_iterator(
        tokenized_titles,
        min_freq=2,
        specials=['<pad>', '<oov>'],
        special_first=True
    )
    target_vocab = build_vocab_from_iterator(
        tokenized_titles,
        min_freq=2
    )

    # Get Feature & Target
    input_sequences = ngram_generate(tokenized_titles, features_vocab, target_vocab)
    X = [sequence[:-1] for sequence in input_sequences]
    y = [sequence[-1] for sequence in input_sequences]

    # Padding & One-Hot Encoding
    longest_sequence_feature = max(len(sequence) for sequence in X)
    padded_X = [F.pad(torch.tensor(sequence), (longest_sequence_feature - len(sequence), 0), value=0) for sequence in X]
    padded_X = torch.stack(padded_X)
    y = torch.tensor(y)
    y_one_hot = one_hot(y, num_classes=len(target_vocab))

    # Train-Test Split
    data = TensorDataset(padded_X, y_one_hot)
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    batch_size = 32
    train_data, test_data = random_split(data, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, len(features_vocab), len(target_vocab), longest_sequence_feature


# Model Definition
class LSTM(nn.Module):
    def __init__(self, features_vocab_total_words, target_vocab_total_words, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(features_vocab_total_words, embedding_dim) # Embedding Layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True) # LSTM Layer
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, target_vocab_total_words) # Fully Connected Layer

    def forward(self, x):
        x = x.to(self.embedding.weight.device)
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out[:, -1, :])
        return output

def load_model(features_vocab_total_words, target_vocab_total_words, longest_sequence_feature):
    return LSTM(features_vocab_total_words, target_vocab_total_words, longest_sequence_feature, 200).to(DEVICE)


# Training Process
def calculate_topk_accuracy(model, data_loader, k=3):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            output = model(batch_x) # Forward pass
            _, predicted_indices = output.topk(k, dim=1) # Get top-k predictions
            correct_predictions += torch.any(predicted_indices == torch.argmax(batch_y, dim=1, keepdim=True), dim=1).sum().item()
            total_predictions += batch_y.size(0)

    accuracy = correct_predictions / total_predictions
    return accuracy


def train(model, train_loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0009) # can switch to SGD, tune lr
    print("Training...")
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, torch.max(batch_y, 1)[1])
            loss.backward()
            optimizer.step()
        accuracy = calculate_topk_accuracy(model, train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Train K-Accuracy: {accuracy * 100:.2f}%')


# Testing Process
def test(model, test_loader, k=3):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for batch_x, label in test_loader:
            batch_x, label = batch_x.to(DEVICE), label.to(DEVICE)
            output = model(batch_x)
            loss += criterion(output, torch.max(label, 1)[1]).item()
            total += label.size(0)
            _, predicted_indices = output.topk(k, dim=1)
            correct += torch.any(predicted_indices == torch.argmax(label, dim=1, keepdim=True), dim=1).sum().item()
    return loss / len(test_loader.dataset), correct / total


if __name__ == "__main__":
    train_loader, test_loader, features_vocab_total_words, target_vocab_total_words, longest_sequence_feature = load_data()
    lstm = load_model(features_vocab_total_words, target_vocab_total_words, longest_sequence_feature)
    train(lstm, train_loader, epochs=50)
    accuracy = test(lstm, test_loader)
    print(f"Accuracy: {accuracy:.3f}")
