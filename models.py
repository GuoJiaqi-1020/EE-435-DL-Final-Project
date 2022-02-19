from turtle import forward
import torch
from torch import nn

class FFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        x = self.linear(x)
        prob = nn.Softmax(dim=1)(x)
        y_pred = prob.argmax(1)
        return y_pred

class RNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings = 33271, embedding_dim = 100)
        self.rnn = nn.RNN(input_size = 100, hidden_size = 256, num_layers = 2, nonlinearity = 'tanh', batch_first = True)
        self.fc = nn.Linear(256, 33271) 
        
    def forward(self, x, h=None):
        x = self.embedding(x.squeeze())
        if h is None:
            x, h = self.rnn(x)
        else:
            x, h = self.rnn(x, h)
        x = self.fc(x)
        y_prob = nn.Softmax(dim=2)(x)
        return y_prob, h