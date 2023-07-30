import numpy as np
import torch.cuda
import torch.nn as nn
import torch

import pyDOE
from pyDOE import lhs
from torch import from_numpy

import matplotlib.pyplot as plt



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
    
    if RNG_key != None:
        np.random.seed(RNG_key)
        pass
    
    
    # normal condition / interior points
    n_st_train = np.concatenate([np.random.uniform(*t_range, (num_nc, 1)), 
                        np.random.uniform(*S_range, (num_nc, 1))], axis=1)
    n_v_train = np.zeros((num_nc, 1))
    

    # final condition (t = T, S is randomized)
    f_st = np.concatenate([np.ones((num_fc, 1)),
                    np.random.uniform(*S_range, (num_fc, 1))], axis=1)
    f_v = gs(f_st[:, 1]).reshape(-1, 1)
    
    # lower boundary condition (S = 0, t is randomized)
    lb_st = np.concatenate([np.random.uniform(*t_range, (num_bc, 1)),
                        0 * np.ones((num_bc, 1))], axis=1)
    lb_v = np.zeros((num_bc, 1))
    
    # upper boundary condition (S = Smax, t is randomized)
    ub_st = np.concatenate([np.random.uniform(*t_range, (num_bc, 1)), 
                        Smax * np.ones((num_bc, 1))], axis=1)
    ub_v = (Smax - K*np.exp(-r*(T-ub_st[:, 0].reshape(-1)))).reshape(-1, 1)
    
    # append boundary condition training points (edge points)
    bc_st_train = np.vstack([f_st, lb_st, ub_st])
    bc_v_train = np.vstack([f_v, lb_v, ub_v])
    
    # Generate the training labels
    all_st_train = np.vstack((n_st_train, bc_st_train)) # append training points to collocation points
    
    return all_st_train, bc_st_train, bc_v_train, n_st_train, n_v_train




def testingData(lb, ub, u, f, num):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  X=np.linspace(lb[0], ub[0], num)
  Y=np.linspace(lb[1], ub[1], num)
    
  X, Y = np.meshgrid(X,Y) #X, Y are (256, 256) matrices

  U = u(X,Y)
  u_test = U.flatten('F')[:,None]
  u_test = torch.from_numpy(u_test).to(device)
    
  xy_test = np.hstack((X.flatten('F')[:,None], Y.flatten('F')[:,None]))
  f_test = f(xy_test[:,[0]], xy_test[:,[1]])
  f_test = torch.from_numpy(f_test).to(device)

  x_test = torch.from_numpy(xy_test[:,[0]]).to(device)
  y_test = torch.from_numpy(xy_test[:,[1]]).to(device)
#   f_test = f(x_test, y_test)
  return x_test, y_test, xy_test, u_test, f_test, X, Y, U

