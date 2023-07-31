# This is supposed to be a template for how to implement PINNS models for the code
import torch
import torch.nn as nn
import torch.nn.functional as F
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








# Locally adaptive neural network
class AdaptiveLinear(nn.Linear):
    r"""Applies a linear transformation to the input data as follows
    :math:`y = naxA^T + b`.
    More details available in Jagtap, A. D. et al. Locally adaptive
    activation functions with slope recovery for deep and
    physics-informed neural networks, Proc. R. Soc. 2020.

    Parameters
    ----------
    in_features : int
        The size of each input sample
    out_features : int 
        The size of each output sample
    bias : bool, optional
        If set to ``False``, the layer will not learn an additive bias
    adaptive_rate : float, optional
        Scalable adaptive rate parameter for activation function that
        is added layer-wise for each neuron separately. It is treated
        as learnable parameter and will be optimized using a optimizer
        of choice
    adaptive_rate_scaler : float, optional
        Fixed, pre-defined, scaling factor for adaptive activation
        functions
    """
    def __init__(self, in_features, out_features, bias=True, adaptive_rate=None, adaptive_rate_scaler=None):
        super(AdaptiveLinear, self).__init__(in_features, out_features, bias)
        self.adaptive_rate = adaptive_rate
        self.adaptive_rate_scaler = adaptive_rate_scaler
        if self.adaptive_rate:
            self.A = nn.Parameter(self.adaptive_rate * torch.ones(self.in_features))
            if not self.adaptive_rate_scaler:
                self.adaptive_rate_scaler = 10.0
            
    def forward(self, input):
        if self.adaptive_rate:
            return nn.functional.linear(self.adaptive_rate_scaler * self.A * input, self.weight, self.bias)
        return nn.functional.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, '
            f'adaptive_rate={self.adaptive_rate is not None}, adaptive_rate_scaler={self.adaptive_rate_scaler is not None}'
        )


class ImprovedNeuralNetwork(nn.Module):
    r"""Neural approximator for the unknown function that is supposed
    to be solved.

    More details available in Raissi, M. et al. Physics-informed neural
    networks: A deep learning framework for solving forward and inverse
    problems involving nonlinear partial differential equations, J.
    Comput. Phys. 2019.

    Parameters
    ----------
    sizes : list
        Each element represents the number of neuron per layer
    activation : callable 
        Activation function
    dropout_rate : float, optional
        Dropout rate for regulrization during training process and
        uncertainty quantification by means of Monte Carlo dropout
        procedure while performing evaluation
    adaptive_rate : float, optional
        Scalable adaptive rate parameter for activation function that
        is added layer-wise for each neuron separately. It is treated
        as learnable parameter and will be optimized using a optimizer
        of choice
    adaptive_rate_scaler : float, optional
        Fixed, pre-defined, scaling factor for adaptive activation
        functions
    """
    def __init__(self, sizes, activation, dropout_rate=0.0, adaptive_rate=None, adaptive_rate_scaler=None):
        super(ImprovedNeuralNetwork, self).__init__()
        self.regressor = nn.Sequential(
            *[ImprovedNeuralNetwork.linear_block(in_features, out_features, activation, dropout_rate, adaptive_rate, adaptive_rate_scaler)
            for in_features, out_features in zip(sizes[:-1], sizes[1:-1])],     
            AdaptiveLinear(sizes[-2], sizes[-1]) # output layer is regular linear transformation
            )
        
    def forward(self, x):
        return self.regressor(x)

    @staticmethod
    def linear_block(in_features, out_features, activation, dropout_rate, adaptive_rate, adaptive_rate_scaler):
        activation_dispatcher = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU()],
            ['relu', nn.ReLU()],
            ['tanh', nn.Tanh()],
            ['sigmoid', nn.Sigmoid()],
            # ['swish', Swish()]
        ])
        return nn.Sequential(
            AdaptiveLinear(in_features, out_features, adaptive_rate=adaptive_rate, adaptive_rate_scaler=adaptive_rate_scaler),
            activation_dispatcher[activation],
            nn.Dropout(dropout_rate),
            )





# # Deep Garlerkin Method (Neural Network)
# # The deep garlerkin method from paper
# class DGMCell(nn.Module):
#   def __init__(self, input_dim, hidden_dim, output_dim,  n_layers):
#     super(DGMCell, self).__init__()
#     self.input_dim = input_dim
#     self.hidden_dim = hidden_dim
#     self.output_dim = output_dim
#     self.n = n_layers

#     self.sig_act = nn.Tanh()

#     self.Sw = nn.Linear(self.input_dim, self.hidden_dim)

#     self.Uz = nn.Linear(self.input_dim, self.hidden_dim)
#     self.Wsz = nn.Linear(self.hidden_dim, self.hidden_dim)

#     self.Ug = nn.Linear(self.input_dim, self.hidden_dim)
#     self.Wsg = nn.Linear(self.hidden_dim, self.hidden_dim)

#     self.Ur = nn.Linear(self.input_dim, self.hidden_dim)
#     self.Wsr = nn.Linear(self.hidden_dim, self.hidden_dim)
    
#     self.Uh = nn.Linear(self.input_dim, self.hidden_dim)
#     self.Wsh = nn.Linear(self.hidden_dim, self.hidden_dim)

#     self.Wf = nn.Linear(hidden_dim, output_dim)
    

#   def forward(self, x):
#     S1 = self.Sw(x)
#     for i in range(self.n):
#       if i==0:
#         S = S1
#       else:
#         S = self.sig_act(out)
#       Z = self.sig_act(self.Uz(x) + self.Wsz(S))
#       G = self.sig_act(self.Ug(x) + self.Wsg(S1))
#       R = self.sig_act(self.Ur(x) + self.Wsr(S))
#       H = self.sig_act(self.Uh(x) + self.Wsh(S*R))
#       out = (1-G)*H + Z*S
#     out = self.Wf(out)
#     return out



# class AdaptiveReluFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, slopes):
#         ctx.save_for_backward(input, slopes)
#         output = nn.ReLU(input) * slopes
#         return output
#     @staticmethod
#     def backward(ctx, grad_output):
#         input, slopes = ctx.saved_tensors
#         grad_input = grad_output * (input > 0).float() * slopes
#         grad_slopes = (grad_output * (input > 0).float() * input).sum(dim=0)
#         return grad_input, grad_slopes

# class AdaptiveRelu(nn.Module):
#     def __init__(self, num_neurons):
#         super(AdaptiveRelu, self).__init__()
#         self.slopes = nn.Parameter(torch.ones(num_neurons))

#     def forward(self, input):
#         return AdaptiveReluFunction.apply(input, self.slopes)


# # Define the feedforward neural network
# class ImprovedFNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, n_layers):
#         super(ImprovedFNN, self).__init__()
#         self.input = nn.Linear(input_size, hidden_size)
#         self.hidden = nn.ModuleList([
#           AdaptiveRelu(hidden_size) for i in range(n_layers-1)
#           ])
#         self.output = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = self.input(x)
#         for layer in self.hidden:
#             x = layer(x)
#         x = self.output(x)
#         return x
