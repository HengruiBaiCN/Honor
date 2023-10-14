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


# This file generates training data
def trainingData(K, r, sigma, T, Smax, S_range, t_range, gs, num_bc, num_fc, num_nc, RNG_key=None):
    '''
    @param num_bc: number of points on the boundary condition
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # set random seed
    if RNG_key != None:
        np.random.seed(RNG_key)
        pass
    
    # normal condition / interior points
    n_st_train = np.concatenate([np.random.uniform(*t_range, (num_nc, 1)), 
                      np.random.uniform(*S_range, (num_nc, 1))], axis=1)
    # n_v_train = np.zeros((num_nc, 1))
    

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
    
    # save training data points to tensor and send to device
    n_st_train = torch.from_numpy(n_st_train).float().requires_grad_().to(device)
            
    bc_st_train = torch.from_numpy(bc_st_train).float().to(device)
    bc_v_train = torch.from_numpy(bc_v_train).float().to(device)
    
    return bc_st_train, bc_v_train, n_st_train



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

    X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


def network_dispatcher(net, sizes, activation, dropout_rate, adaptive_rate, adaptive_rate_scaler):
    """
    Return network from `SUPPORTED_NETWORKS`.
    """
    assert isinstance(net, (str, )), '`network` type must be str.'
    net = net.lower()
    if net == 'pinn':
        return networks.FeedforwardNeuralNetwork(sizes[0], sizes[-2], sizes[-1], len(sizes)-2)
    elif net == 'ipinn':
        return networks.ImprovedNeuralNetwork(
            sizes, activation, dropout_rate, adaptive_rate, adaptive_rate_scaler                                    )
    else:
        return 'Incorrect neural network input'


def loss_dispatcher(pde_loss, bc_loss, data_loss, slope_recovery_term, loss_weights, adaptive_weight, x_f_s, x_label_s, x_data_s):
    '''
    @param adaptive_rate: bool, whether to use adaptive rate or not
    @param model: model, used to get the local recovery term
    @param x_f_s: weight for pde residual loss
    @param x_label_s: weight for bc residual loss
    @param x_data_s: weight for data residual loss
    @param adaptive_weight: adaptive weight or not
    @return: loss
    '''
    loss = None
    if slope_recovery_term and adaptive_weight:
        loss = torch.exp(-x_f_s.detach()) * pde_loss + torch.exp(-x_label_s.detach()) * bc_loss + torch.exp(-x_data_s.detach()) * data_loss + x_f_s + x_label_s + x_data_s + slope_recovery_term
    elif slope_recovery_term and not adaptive_weight:
        loss = loss_weights[0] * pde_loss + loss_weights[1] * bc_loss + loss_weights[2] * data_loss + slope_recovery_term
    elif adaptive_weight and not slope_recovery_term:
        loss = torch.exp(-x_f_s.detach()) * pde_loss + torch.exp(-x_label_s.detach()) * bc_loss + torch.exp(-x_data_s.detach()) * data_loss + x_f_s + x_label_s + x_data_s
    else:
        loss = loss_weights[0] * pde_loss + loss_weights[1] * bc_loss + loss_weights[2] * data_loss
    return loss



def network_training(
        K, r, sigma, T, Smax, S_range, t_range, gs, num_bc, num_fc, num_nc, RNG_key,
        device, net, sizes, activation, learning_rate, aw_learning_rate, n_epochs, lossFunction,
        dropout_rate, adaptive_rate, adaptive_rate_scaler, loss_weights, adaptive_weight,
        X_train_tensor, y_train_tensor
        ):
    r"""Train PINN and return trained network alongside loss over time.

    Parameters
    ----------
    device : str
        Specifiy `cuda` if CUDA-enabled GPU is available, otherwise
        specify `cpu`
    sizes : list
        Each element represents the number of neuron per layer
    activation : callable 
        Activation function
    n_epochs : int
        The number of training epochs
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # adaptive weight
    if net == 'pinn':
        x_f_s = torch.tensor(-1.).float().to(device).requires_grad_(True)
        x_label_s = torch.tensor(0.).float().to(device).requires_grad_(True)
        x_data_s = torch.tensor(0.).float().to(device).requires_grad_(True)
        optimizer_adam_weight = torch.optim.Adam([x_f_s] + [x_label_s] + [x_data_s], lr=aw_learning_rate)
    else:
        x_f_s = torch.tensor(0.).float().to(device).requires_grad_(True)
        x_label_s = torch.tensor(0.).float().to(device).requires_grad_(True)
        x_data_s = torch.tensor(-1.).float().to(device).requires_grad_(True)
        optimizer_adam_weight = torch.optim.Adam([x_f_s] + [x_label_s] + [x_data_s], lr=aw_learning_rate)
    
    # record loss history for plotting and save the best model
    mse_loss_hist = []
    pde_loss_hist = []
    bc_loss_hist = []
    data_loss_hist = []
    x_f_s_hist = []
    x_label_s_hist = []
    x_data_s_hist = []
    loss_weights_hist = {}
    min_train_loss = float("inf")  # Initialize with a large value
    final_model = None
    
    # training loop and logging setup
    logging.info(f'{model}\n')
    logging.info(f'Training started at {datetime.datetime.now()}\n')
    start_time = timer()
    # training loop
    for _ in tqdm.tqdm(range(n_epochs), desc='[Training procedure]', ascii=True, total=n_epochs):

        # sampling
        bc_st_train, bc_v_train, n_st_train = \
            trainingData(K, r, sigma, T, Smax, S_range, t_range, gs, num_bc, num_fc, num_nc, RNG_key)
        
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
        y2_hat = model(bc_st_train)
        bc_loss = lossFunction(bc_v_train, y2_hat)
        
        # data Round loss 
        y3_hat = model(X_train_tensor)
        data_loss = lossFunction(y_train_tensor, y3_hat)
        
        # slope recovery term
        slope_recovery_term = 0
        if adaptive_rate:
            local_recovery_terms = torch.tensor([torch.mean(model.regressor[layer][0].A.data) for layer in range(len(model.regressor) - 1)])
            slope_recovery_term = 1 / torch.mean(torch.exp(local_recovery_terms))
        
        # update the model by backpropagation and calculate the total loss
        optimizer.zero_grad()
        loss = loss_dispatcher(pde_loss, bc_loss, data_loss, slope_recovery_term, loss_weights, adaptive_weight, x_f_s, x_label_s, x_data_s)
        loss.backward()
        optimizer.step()
        
        # update the weight if adaptive weight is used by backpropagation 
        if adaptive_weight:
            optimizer_adam_weight.zero_grad()
            if adaptive_rate:
                loss1 = torch.exp(-x_f_s) * pde_loss.detach() + torch.exp(-x_label_s) * bc_loss.detach() + torch.exp(-x_data_s) * data_loss.detach() + x_data_s + x_f_s + x_label_s + slope_recovery_term
                pass
            else:
                loss1 = torch.exp(-x_f_s) * pde_loss.detach() + torch.exp(-x_label_s) * bc_loss.detach() + torch.exp(-x_data_s) * data_loss.detach() + x_data_s + x_f_s + x_label_s
                pass
            loss1.backward()
            optimizer_adam_weight.step()
        
        # record loss history for plotting
        mse_loss = pde_loss + bc_loss + data_loss
        mse_loss_hist.append(mse_loss.item())
        pde_loss_hist.append(pde_loss.item())
        bc_loss_hist.append(bc_loss.item())
        data_loss_hist.append(data_loss.item())
        x_f_s_hist.append(x_f_s.item())
        x_label_s_hist.append(x_label_s.item())
        x_data_s_hist.append(x_data_s.item())
        
        # save the best model by comparing the loss for testing data
        if mse_loss.item() < min_train_loss:
            min_train_loss = mse_loss.item()
            final_model = model.state_dict()
            pass
        pass
    
    # logging setup and training time calculation
    loss_weights_hist['PDE Weight'] = x_f_s_hist
    loss_weights_hist['BC Weight'] = x_label_s_hist
    loss_weights_hist['Data Weight'] = x_data_s_hist
    elapsed = timer() - start_time
    logging.info(f'Training finished. Elapsed time: {elapsed} s\n')
    
    return model, final_model, mse_loss_hist, pde_loss_hist, bc_loss_hist, data_loss_hist, loss_weights_hist



# SUPPORTED_OPTIMIZERS = ['bfgs', 'sgd', 'adam']
# def optimizer_dispatcher(optimizer, parameters, learning_rate):
#     """Return optimization function from `SUPPORTED_OPTIMIZERS`.

#     Parameters
#     ----------
#     optimizer : str
#         Optimization function name
#     parameters : callable
#         Network parameters
#     learning_rate : float
#         Learning rate

#     Returns
#     -------
#     callable
#         Optimization function
#     """
#     assert isinstance(optimizer, (str, )), '`optimizer` type must be str.'
#     optimizer = optimizer.lower()
#     assert optimizer in SUPPORTED_OPTIMIZERS, 'Invalid optimizer. Falling to default.'
#     if optimizer == 'bfgs':
#         return torch.optim.LBFGS(parameters, line_search_fn="strong_wolfe")
#     elif optimizer == 'sgd':
#         return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=1e-2*learning_rate)
#     else:
#         return torch.optim.Adam(parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-5)


# def network_training(
#         K, r, sigma, T, Smax, S_range, t_range, gs, num_bc, num_fc, num_nc, RNG_key,
#         device, net, opt, sizes, activation, learning_rate, n_epochs, lossFunction,
#         dropout_rate, adaptive_rate, adaptive_rate_scaler, loss_weights, adaptive_weight,
#         X_train_tensor, y_train_tensor
#         ):
#     r"""Train PINN and return trained network alongside loss over time.

#     Parameters
#     ----------
#     device : str
#         Specifiy `cuda` if CUDA-enabled GPU is available, otherwise
#         specify `cpu`
#     domain : tuple or list
#         Boundaries of the solution domain
#     boundary_conditions : tuple or list
#         Boundary conditions
#     rhs : float
#         Value of the scalar right hand side function
#     sizes : list
#         Each element represents the number of neuron per layer
#     activation : callable 
#         Activation function
#     optimizer : callable
#         Optimization procedure
#     n_epochs : int
#         The number of training epochs
#     batch_size : int
#         The number of data points for optimization per epoch
#     linspace : bool
#         Space the batch of data linearly, otherwise random
#     learning_rate : float
#         Learning rate
#     dropout_rate : float, optional
#         Dropout rate for regulrization during training process and
#         uncertainty quantification by means of Monte Carlo dropout
#         procedure while performing evaluation
#     adaptive_rate : float, optional
#         Scalable adaptive rate parameter for activation function that
#         is added layer-wise for each neuron separately. It is treated
#         as learnable parameter and will be optimized using a optimizer
#         of choice
#     adaptive_rate_scaler : float, optional
#         Fixed, pre-defined, scaling factor for adaptive activation
#         functions
    
#     Returns
#     -------
#     model : Net
#         Trained function approximator
#     loss_list : list
#         Loss values during training process
#     """
    
#     # initialize model and optimizer
#     model = network_dispatcher(net, sizes, activation, dropout_rate, adaptive_rate, adaptive_rate_scaler).to(device=device)
#     optimizer = optimizer_dispatcher(opt, model.parameters(), learning_rate)
    
#     # adaptive weight
#     x_f_s = torch.tensor(0.).float().to(device).requires_grad_(True)
#     x_label_s = torch.tensor(0.).float().to(device).requires_grad_(True)
#     x_data_s = torch.tensor(0.).float().to(device).requires_grad_(True)
#     optimizer_adam_weight = torch.optim.Adam([x_f_s] + [x_label_s] + [x_data_s], lr=learning_rate*10)
    
    
#     # training
#     loss_hist = []
#     logging.info(f'{model}\n')
#     logging.info(f'Training started at {datetime.datetime.now()}\n')
#     min_train_loss = float("inf")  # Initialize with a large value
#     final_model = None
#     start_time = timer()
    
#     # training loop
#     for _ in tqdm.tqdm(range(n_epochs), desc='[Training procedure]', ascii=True, total=n_epochs):

#         # sampling
#         bc_st_train, bc_v_train, n_st_train, n_v_train = \
#             trainingData(K, r, sigma, T, Smax, S_range, t_range, gs, num_bc, num_fc, num_nc, RNG_key)
#         # save training data points to tensor and send to device
#         n_st_train = torch.from_numpy(n_st_train).float().requires_grad_().to(device)
#         n_v_train = torch.from_numpy(n_v_train).float().to(device)
                
#         bc_st_train = torch.from_numpy(bc_st_train).float().to(device)
#         bc_v_train = torch.from_numpy(bc_v_train).float().to(device)
        
#         # pde residual loss
#         y1_hat = model(n_st_train)
#         grads = tgrad.grad(y1_hat, n_st_train, grad_outputs=torch.ones(y1_hat.shape).cuda(), 
#                     retain_graph=True, create_graph=True, only_inputs=True)[0]
#         dVdt, dVdS = grads[:, 0].view(-1, 1), grads[:, 1].view(-1, 1)
#         grads2nd = tgrad.grad(dVdS, n_st_train, grad_outputs=torch.ones(dVdS.shape).cuda(), 
#                         create_graph=True, only_inputs=True, allow_unused=True)[0]
#         S1 = n_st_train[:, 1].view(-1, 1)
#         d2VdS2 = grads2nd[:, 1].view(-1, 1)
#         pde_loss = lossFunction(-dVdt, 0.5*((sigma*S1)**2)*d2VdS2 + r*S1*dVdS - r*y1_hat)
        
#         # boudary condition loss
#         y2_hat = model(bc_st_train)
#         bc_loss = lossFunction(bc_v_train, y2_hat)
        
#         # data Round
#         y3_hat = model(X_train_tensor)
#         data_loss = lossFunction(y_train_tensor, y3_hat)
        
#         # calculate the total loss and update the model
#         optimizer.zero_grad()
#         loss = loss_dispatcher(pde_loss, bc_loss, data_loss, adaptive_rate, model, loss_weights, adaptive_weight, x_f_s, x_label_s, x_data_s)
#         loss.backward()
#         optimizer.step()
        
#         # update the weight if adaptive weight is used
#         if adaptive_weight:
#             optimizer_adam_weight.zero_grad()
#             loss1 = torch.exp(-x_f_s) * pde_loss.detach() + x_f_s + torch.exp(-x_label_s) * bc_loss.detach() + x_label_s + torch.exp(-x_data_s) * data_loss.detach() + x_data_s
#             loss1.backward()
#             optimizer_adam_weight.step()
#             pass
        
#         mse_loss = pde_loss + bc_loss + data_loss
#         loss_hist.append(mse_loss.item())
#         if mse_loss.item() < min_train_loss:
#             min_train_loss = mse_loss.item()
#             final_model = model.state_dict()
#             pass
#     elapsed = timer() - start_time
#     logging.info(f'Training finished. Elapsed time: {elapsed} s\n')
    
#     return model, loss_hist, final_model
