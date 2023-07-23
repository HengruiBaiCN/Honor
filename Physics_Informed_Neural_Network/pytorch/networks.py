# This is supposed to be a template for how to implement PINNS models for the code
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import math
import matplotlib.pyplot as plt


# Define the feedforward neural network
class FeedforwardNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(FeedforwardNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        self.layers.extend([nn.Linear(hidden_size, hidden_size) for _ in range(n_layers - 1)])
        self.output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        x = self.output(x)
        return x


# Deep Garlerkin Method (Neural Network)
class DGMCell(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim,  n_layers):
    super(DGMCell, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.n = n_layers

    self.sig_act = nn.Tanh()

    self.Sw = nn.Linear(self.input_dim, self.hidden_dim)

    self.Uz = nn.Linear(self.input_dim, self.hidden_dim)
    self.Wsz = nn.Linear(self.hidden_dim, self.hidden_dim)

    self.Ug = nn.Linear(self.input_dim, self.hidden_dim)
    self.Wsg = nn.Linear(self.hidden_dim, self.hidden_dim)

    self.Ur = nn.Linear(self.input_dim, self.hidden_dim)
    self.Wsr = nn.Linear(self.hidden_dim, self.hidden_dim)
    
    self.Uh = nn.Linear(self.input_dim, self.hidden_dim)
    self.Wsh = nn.Linear(self.hidden_dim, self.hidden_dim)

    self.Wf = nn.Linear(hidden_dim, output_dim)
    

  def forward(self, x):
    S1 = self.Sw(x)
    for i in range(self.n):
      if i==0:
        S = S1
      else:
        S = self.sig_act(out)
      Z = self.sig_act(self.Uz(x) + self.Wsz(S))
      G = self.sig_act(self.Ug(x) + self.Wsg(S1))
      R = self.sig_act(self.Ur(x) + self.Wsr(S))
      H = self.sig_act(self.Uh(x) + self.Wsh(S*R))
      out = (1-G)*H + Z*S
    out = self.Wf(out)
    return out



# The Discriminator Network
# A general discriminator. Input_size is the physical dimension of the problem, output_size the dimension of the residual
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map = nn.Sequential(
            nn.Linear(input_size, 2 * hidden_size),
            nn.ReLU(),
#             nn.Tanh(),
            nn.Linear(2 * hidden_size, 2 * hidden_size),
            nn.ReLU(),
#             nn.Tanh(),
            nn.Linear(2 * hidden_size, 2 * hidden_size),
            nn.ReLU(),
#             nn.Tanh(),
            nn.Linear(2 * hidden_size, 2 * hidden_size),
            nn.ReLU(),
#             nn.Tanh(),
            nn.Linear(2  * hidden_size, output_size),
        )

    def forward(self, x, y):
        return self.map(torch.cat((x,y), 1))