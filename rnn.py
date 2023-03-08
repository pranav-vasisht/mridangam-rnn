import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

class Encoder:
    def __init__(self, language_file_path):
        self.word_to_index = {}
        self.index_to_word = {}
        self.num_words = 0
        self.load_language(language_file_path)

    def load_language(self, language_file_path):
        with open(language_file_path, 'r') as f:
            for line in f:
                for word in line.split():
                    if word not in self.word_to_index:
                        self.word_to_index[word] = self.num_words
                        self.index_to_word[self.num_words] = word
                        self.num_words += 1

    def encode(self, sentence):
        encoding = []
        for word in sentence.split():
            if word in self.word_to_index:
                encoding.append(self.word_to_index[word])
            else:
                encoding.append(self.word_to_index['<UNK>'])
        return torch.tensor(encoding)

# Example usage:

input_size = 20000  # Replace with appropriate size for encoded input
hidden_size = 500000
output_size = 10000
seq_length = 6

print(input_size)
print(hidden_size)
print(output_size)
print(seq_length)

language_file_path = 'new_language.txt'
encoder = Encoder(language_file_path)

rnn = RNN(input_size, hidden_size, output_size)
print(rnn)

# Initialize hidden state
hidden = rnn.initHidden()
print("initHidden done")

# Generate a prior sequence of inputs
prior_sequence = 'This is a test sequence'
encoded_sequence = encoder.encode(prior_sequence)
inputs = [encoded_sequence[i] for i in range(len(encoded_sequence))]

print(prior_sequence)
print(encoded_sequence)
print(inputs)

# Feed the prior sequence through the RNN to obtain a hidden state
for i in range(len(encoded_sequence)):
    output, hidden = rnn(inputs[i], hidden)

# Use the hidden state to generate new outputs
new_sequence = 'This is a new test sequence'
encoded_new_sequence = encoder.encode(new_sequence)
new_outputs = []
for i in range(len(encoded_new_sequence)):
    output, hidden = rnn(encoded_new_sequence[i], hidden)
    new_outputs.append(output)

print(new_outputs)
