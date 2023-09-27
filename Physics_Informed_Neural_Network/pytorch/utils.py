import torch
import torch.cuda
import torch.nn as nn
from torch import from_numpy
import torch.autograd as tgrad


import pyDOE
import numpy as np
import pandas as pd
from pyDOE import lhs


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tqdm

import argparse
import datetime
import logging
import os
import sys
import networks
from timeit import default_timer as timer



def printMemory():
  t = torch.cuda.get_device_properties(0).total_memory
  r = torch.cuda.memory_reserved(0)
  a = torch.cuda.memory_allocated(0)
  f = r-a  # free inside reserved
  print(f"total: {t}, reserved: {r}, free: {f}")
  
  
def trainingData3(K, r, sigma, T, Smax, S_range, t_range, gs, num_bc, num_fc, num_nc, RNG_key=None):
    '''
    @param num_bc: number of points on the boundary condition
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if RNG_key != None:
        np.random.seed(RNG_key)
        pass
    
    
    # normal condition / interior points
    n_st_train = np.concatenate([np.random.uniform(*t_range, (num_nc, 1)), 
                      np.random.uniform(*S_range, (num_nc, 1))], axis=1)
    n_v_train = np.zeros((num_nc, 1))
    

    # final condition (t = 0, S is randomized)
    i_st_train = np.concatenate([np.ones((num_fc, 1)),
                    np.random.uniform(*S_range, (num_fc, 1))], axis=1)
    i_v_train = gs(i_st_train[:, 1]).reshape(-1, 1)
    
    # lower boundary condition (S = 0, t is randomized)
    lb_st = np.concatenate([np.random.uniform(*t_range, (num_bc, 1)),
                        0 * np.ones((num_bc, 1))], axis=1)
    lb_v = np.zeros((num_bc, 1))
    
    # upper boundary condition (S = Smax, t is randomized)
    ub_st = np.concatenate([np.random.uniform(*t_range, (num_bc, 1)), 
                        Smax * np.ones((num_bc, 1))], axis=1)
    ub_v = (Smax - K*np.exp(-r*(T-ub_st[:, 0].reshape(-1)))).reshape(-1, 1)
    
    # append boundary condition training points (edge points)
    bc_st_train = np.vstack([lb_st, ub_st])
    bc_v_train = np.vstack([lb_v, ub_v])
    
    
    return i_st_train, i_v_train, bc_st_train, bc_v_train, n_st_train, n_v_train



# This file generates training data
def trainingData(K, r, sigma, T, Smax, S_range, t_range, gs, num_bc, num_fc, num_nc, RNG_key=None):
    '''
    @param num_bc: number of points on the boundary condition
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if RNG_key != None:
        np.random.seed(RNG_key)
        pass
    
    
    # normal condition / interior points
    n_st_train = np.concatenate([np.random.uniform(*t_range, (num_nc, 1)), 
                      np.random.uniform(*S_range, (num_nc, 1))], axis=1)
    n_v_train = np.zeros((num_nc, 1))
    

    # initial condition (t = T, S is randomized)
    i_st_train = np.concatenate([np.ones((num_fc, 1)),
                    np.random.uniform(*S_range, (num_fc, 1))], axis=1)
    i_v_train = gs(i_st_train[:, 1]).reshape(-1, 1)
    
    # lower boundary condition (S = 0, t is randomized)
    lb_st = np.concatenate([np.random.uniform(*t_range, (num_bc, 1)),
                        0 * np.ones((num_bc, 1))], axis=1)
    lb_v = np.zeros((num_bc, 1))
    
    # upper boundary condition (S = Smax, t is randomized)
    ub_st = np.concatenate([np.random.uniform(*t_range, (num_bc, 1)), 
                        Smax * np.ones((num_bc, 1))], axis=1)
    ub_v = (Smax - K*np.exp(-r*(T-ub_st[:, 0].reshape(-1)))).reshape(-1, 1)
    
    # append boundary condition training points (edge points)
    bc_st_train = np.vstack([lb_st, ub_st, i_st_train])
    bc_v_train = np.vstack([lb_v, ub_v, i_v_train])
    
    
    return bc_st_train, bc_v_train, n_st_train, n_v_train


def fdm_data(Smax, T, M, N, src, device):
    data = pd.read_csv(src)
    
    ds = Smax/M
    dt = T / N

    input_data = []
    output_data = []
    for row_index, row in data.iterrows():
        for col_index, value in enumerate(row):
            input_data.append([(N-row_index)*dt, col_index*ds])  # Store row and column index as input
            output_data.append([value])  # Store the corresponding value as output
    # X = torch.tensor(input_data, dtype=torch.float32).to(device)
    # y = torch.tensor(output_data, dtype=torch.float32).to(device)

    X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor





SUPPORTED_OPTIMIZERS = ['bfgs', 'sgd', 'adam']
def optimizer_dispatcher(optimizer, parameters, learning_rate):
    r"""Return optimization function from `SUPPORTED_OPTIMIZERS`.

    Parameters
    ----------
    optimizer : str
        Optimization function name
    parameters : callable
        Network parameters
    learning_rate : float
        Learning rate

    Returns
    -------
    callable
        Optimization function
    """
    assert isinstance(optimizer, (str, )), '`optimizer` type must be str.'
    optimizer = optimizer.lower()
    assert optimizer in SUPPORTED_OPTIMIZERS, 'Invalid optimizer. Falling to default.'
    if optimizer == 'bfgs':
        return torch.optim.LBFGS(parameters, line_search_fn="strong_wolfe")
    elif optimizer == 'sgd':
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=1e-2*learning_rate)
    else:
        return torch.optim.Adam(parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-5)


def network_dispatcher(net, sizes, activation, dropout_rate, adaptive_rate, adaptive_rate_scaler):
    r"""Return optimization function from `SUPPORTED_OPTIMIZERS`.

    Parameters
    ----------
    optimizer : str
        Optimization function name
    parameters : callable
        Network parameters
    learning_rate : float
        Learning rate

    Returns
    -------
    callable
        Optimization function
    """
    assert isinstance(net, (str, )), '`optimizer` type must be str.'
    net = net.lower()
    # assert net in SUPPORTED_OPTIMIZERS, 'Invalid optimizer. Falling to default.'
    if net == 'pinn':
        return networks.FeedforwardNeuralNetwork(sizes[0], sizes[-2], sizes[-1], len(sizes))
    elif net == 'ipinn':
        return networks.ImprovedNeuralNetwork(
            sizes, activation, dropout_rate, adaptive_rate, adaptive_rate_scaler                                    )
    else:
        return 'Incorrect neural network input'


def loss_dispatcher(pde_loss, bc_loss, adaptive_rate, model, w1, w2, adaptive_weight, x_f_s, x_label_s):
    '''
    @param adaptive_rate: bool, whether to use adaptive rate or not
    @param model: model, used to get the local recovery term
    @param w1: weight for pde loss
    @param w2: weight for bc loss
    @param adaptive_weight: bool, whether to use adaptive weight or not
    @return: loss
    '''
    loss = None
    if adaptive_rate:
        local_recovery_terms = torch.tensor([torch.mean(model.regressor[layer][0].A.data) for layer in range(len(model.regressor) - 1)])
        slope_recovery_term = 1 / torch.mean(torch.exp(local_recovery_terms))
        loss = w1 * pde_loss + w2 * bc_loss + slope_recovery_term
    elif adaptive_weight:
        loss = torch.exp(-x_f_s.detach()) * pde_loss + torch.exp(-x_label_s.detach()) * bc_loss
    else:
        loss = w1 * pde_loss + w2 * bc_loss
    return loss


def network_training(
        K, r, sigma, T, Smax, S_range, t_range, gs, num_bc, num_fc, num_nc, RNG_key,
        device, net, opt, sizes, activation, learning_rate, n_epochs, lossFunction,
        dropout_rate, adaptive_rate, adaptive_rate_scaler, w1, w2, adaptive_weight,
        ):
    r"""Train PINN and return trained network alongside loss over time.

    Parameters
    ----------
    device : str
        Specifiy `cuda` if CUDA-enabled GPU is available, otherwise
        specify `cpu`
    domain : tuple or list
        Boundaries of the solution domain
    boundary_conditions : tuple or list
        Boundary conditions
    rhs : float
        Value of the scalar right hand side function
    sizes : list
        Each element represents the number of neuron per layer
    activation : callable 
        Activation function
    optimizer : callable
        Optimization procedure
    n_epochs : int
        The number of training epochs
    batch_size : int
        The number of data points for optimization per epoch
    linspace : bool
        Space the batch of data linearly, otherwise random
    learning_rate : float
        Learning rate
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
    
    Returns
    -------
    model : Net
        Trained function approximator
    loss_list : list
        Loss values during training process
    """
    
    # initialize model and optimizer
    model = network_dispatcher(net, sizes, activation, dropout_rate, adaptive_rate, adaptive_rate_scaler).to(device=device)
    optimizer = optimizer_dispatcher(opt, model.parameters(), learning_rate)
    
    # adaptive weight
    x_f_s = torch.tensor(0.).float().to(device).requires_grad_(True)
    x_label_s = torch.tensor(0.).float().to(device).requires_grad_(True)
    optimizer_adam_weight = torch.optim.Adam([x_f_s] + [x_label_s], lr=0.0003)
    
    
    # training
    loss_hist = []
    logging.info(f'{model}\n')
    logging.info(f'Training started at {datetime.datetime.now()}\n')
    min_train_loss = float("inf")  # Initialize with a large value
    final_model = None
    start_time = timer()
    
    # training loop
    for _ in tqdm.tqdm(range(n_epochs), desc='[Training procedure]', ascii=True, total=n_epochs):

        # sampling
        bc_st_train, bc_v_train, n_st_train, n_v_train = \
            trainingData(K, r, sigma, T, Smax, S_range, t_range, gs, num_bc, num_fc, num_nc, RNG_key)
        # save training data points to tensor and send to device
        n_st_train = torch.from_numpy(n_st_train).float().requires_grad_().to(device)
        n_v_train = torch.from_numpy(n_v_train).float().to(device)
                
        bc_st_train = torch.from_numpy(bc_st_train).float().to(device)
        bc_v_train = torch.from_numpy(bc_v_train).float().to(device)
        
        # pde residual loss
        y1_hat = model(n_st_train)
        grads = tgrad.grad(y1_hat, n_st_train, grad_outputs=torch.ones(y1_hat.shape).cuda(), 
                    retain_graph=True, create_graph=True, only_inputs=True)[0]
        dVdt, dVdS = grads[:, 0].view(-1, 1), grads[:, 1].view(-1, 1)
        grads2nd = tgrad.grad(dVdS, n_st_train, grad_outputs=torch.ones(dVdS.shape).cuda(), 
                        create_graph=True, only_inputs=True, allow_unused=True)[0]
        S1 = n_st_train[:, 1].view(-1, 1)
        d2VdS2 = grads2nd[:, 1].view(-1, 1)
        pde_loss = lossFunction(-dVdt, 0.5*((sigma*S1)**2)*d2VdS2 + r*S1*dVdS - r*y1_hat)
        
        # boudary condition loss
        y21_hat = model(bc_st_train)
        bc_loss = lossFunction(bc_v_train, y21_hat)
        
        
        loss = loss_dispatcher(pde_loss, bc_loss, adaptive_rate, model, w1, w2, adaptive_weight, x_f_s, x_label_s)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        mse_loss = pde_loss + bc_loss
        loss_hist.append(mse_loss.item())
        
        if adaptive_weight:
            # update the weight
            optimizer_adam_weight.zero_grad()
            loss = torch.exp(-x_f_s) * pde_loss.detach() + x_f_s + torch.exp(-x_label_s) * bc_loss.detach() + x_label_s
            loss.backward()
            optimizer_adam_weight.step()
            pass
        
        if mse_loss.item() < min_train_loss:
            min_train_loss = mse_loss.item()
            final_model = model
            pass
        
        
    elapsed = timer() - start_time
    logging.info(f'Training finished. Elapsed time: {elapsed} s\n')
    return final_model, loss_hist



# def test(device, model):
#     data = pd.read_csv("sample.csv")
#     # Create a mapping from column names to integers
#     # column_to_int_mapping = {col: idx for idx, col in enumerate(data.columns)}
    
#     # the size of each grid
#     ds = 250/200
#     dt = 1 / 10000


#     input_data = []
#     output_data = []
#     for row_index, row in data.iterrows():
#         for col_index, value in enumerate(row):
#             input_data.append([(10000-row_index)*dt, col_index*ds])  # Store row and column index as input
#             output_data.append([value])  # Store the corresponding value as output
#     X = torch.tensor(input_data, dtype=torch.float32).to(device)
#     y = torch.tensor(output_data, dtype=torch.float32).to(device)
    
#     lossfunction = nn.MSELoss()
#     prediction = model(X)
    
#     print(prediction[0])
#     print(prediction[1])
#     print(y[0])
#     print(y[1])
    
#     return lossfunction(prediction, y).item()
