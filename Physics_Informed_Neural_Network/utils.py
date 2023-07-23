import numpy as np
import torch.cuda

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
    
    
    
    # save training data points to tensor and send to device
    all_st_train = torch.from_numpy(all_st_train).float().requires_grad_().to(device) 
    
    n_st_train = torch.from_numpy(n_st_train).float().requires_grad_().to(device)
    n_v_train = torch.from_numpy(n_v_train).float().to(device)
    
    bc_st_train = torch.from_numpy(bc_st_train).float().to(device)
    bc_v_train = torch.from_numpy(bc_v_train).float().to(device)
    
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

def PINNplot(PINN, X, U, info):
    error_vec, u_pred = PINN.test(True)
    
    s=[1 for i in range(len(X[0]))]

    print('Test Error: %.5f'  % (error_vec))

    fig, (plts1, plts2) = plt.subplots(2, 5, figsize = (20,8))
    plts1[0].set_title("at Y=0, X and u_pred")
    plts1[0].set_xlabel("x")
    plts1[0].set_ylabel("u(x, 0)")
    plts1[0].scatter(X[0], u_pred[0], s=s, marker='.', label = "Predicted")
    plts1[0].scatter(X[0], U[0], s=s, marker='.', label = "Exact")
    plts1[0].legend(markerscale=4)

    plts1[1].set_title("at Y=0.25, X and true U")
    plts1[1].set_xlabel("x")
    plts1[1].set_ylabel("u(x, 0.25)")
    plts1[1].scatter(X[24], u_pred[24], s=s, marker='.', label = "Predicted")
    plts1[1].scatter(X[24], U[24], s=s, marker='.', label = "Exact")
    plts1[1].legend(markerscale=4)

    plts1[2].set_title("Iternation and L2 relative error")
    plts1[2].set_xlabel("Iternations")
    plts1[2].set_ylabel("L2 relative error ")
    plts1[2].scatter(info[:,0], info[:,1], marker='.')

    plts1[3].set_title("Iternation and PINN loss")
    plts1[3].set_xlabel("Iternations")
    plts1[3].set_ylabel("PINN loss")
    plts1[3].scatter(info[:,0], info[:,2], marker='.')

    plts1[4].set_title("Iternation and PINN PDE loss")
    plts1[4].set_xlabel("Iternations")
    plts1[4].set_ylabel("PINN PDE loss")
    plts1[4].scatter(info[:,0], info[:,3], marker='.')

    plts2[0].set_title("at Y=0.5, X and u_pred")
    plts2[0].set_xlabel("x")
    plts2[0].set_ylabel("u(x, 0.5)")
    plts2[0].scatter(X[50], u_pred[50], s=s, marker='.', label = "Predicted")
    plts2[0].scatter(X[50], U[50], s=s, marker='.', label = "Exact")
    plts2[0].legend(markerscale=4)

    plts2[1].set_title("at Y=1, X and true U")
    plts2[1].set_xlabel("x")
    plts2[1].set_ylabel("u(x, 1)")
    plts2[1].scatter(X[-1], u_pred[-1], s=s, marker='.', label = "Predicted")
    plts2[1].scatter(X[-1], U[-1], s=s, marker='.', label = "Exact")
    plts2[1].legend(markerscale=4)

    plts2[2].set_title("log10 L2 relative error and iterations")
    plts2[2].set_xlabel("Iternations")
    plts2[2].set_ylabel("log10 L2 relative error ")
    plts2[2].scatter(info[:,0], np.log10(info[:,1]), marker='.')

    plts2[3].set_title("Iternation and log10 PINN loss")
    plts2[3].set_xlabel("Iternations")
    plts2[3].set_ylabel("log10 PINN loss")
    plts2[3].scatter(info[:,0], np.log10(info[:,2]), marker='.')

    plts2[4].set_title("Iternation and log10 PINN PDE loss")
    plts2[4].scatter(info[:,0], np.log10(info[:,3]), marker='.')
    plts2[4].set_xlabel("Iternations")
    plts2[4].set_ylabel("log10 PINN PDE loss")

    fig.tight_layout()
    
    
    
def ACGDPlot(PINN, X, U, info):
    error_vec, u_pred = PINN.test(True)
    s=[1 for i in range(len(X[0]))]

    print('Test Error: %.5f'  % (error_vec))

    fig, (plts1, plts2) = plt.subplots(2, 6, figsize = (24,8))
    
    plts1[0].set_title("at Y=0, X")
    plts1[0].scatter(X[0], u_pred[0], s=s, marker='.', label = "Predicted u")
    plts1[0].scatter(X[0], U[0], s=s, marker='.', label = "Exact u")
    plts1[0].legend()

    plts1[1].set_title("Forward Passes vs iterations")
    plts1[1].scatter(info[:,6], info[:,1], marker='.')
    plts1[1].set_xscale("log")
    plts1[1].set_yscale("log")
    plts1[1].set_xlabel("Forward passes")
    plts1[1].set_ylabel("log10 L2 relative error")

    plts1[2].set_title("log10 L2 error vs log10 iterations")
    plts1[2].scatter(info[:,0], info[:,1], marker='.')
    plts1[2].set_xlim([90, info[-1:,0]+100])
    plts1[2].set_xscale("log")
    plts1[2].set_yscale("log")
    plts1[2].set_xlabel("log10 iterations")
    plts1[2].set_ylabel("log10 L2 relative error")

    plts1[3].set_title("log10 composite loss vs log10 iterations")
    plts1[3].scatter(np.log10(info[:,0]), np.log10(info[:,2]), marker='.')
    plts1[3].set_xlabel("log10 iterations")
    plts1[3].set_ylabel("log10 L2 composite loss")
    
    plts1[4].set_title("log10 PINN loss vs log10 iterations")
    plts1[4].scatter(np.log10(info[:,0]), np.log10(info[:,3]), marker='.')
    plts1[4].set_xlabel("log10 iterations")
    plts1[4].set_ylabel("log10 PINN loss")

    plts1[5].set_title("log10 PINN PDE loss vs log10 iterations")
    plts1[5].scatter(np.log10(info[:,0]), np.log10(info[:,4]), marker='.')
    plts1[5].set_xlabel("log10 iterations")
    plts1[5].set_ylabel("log10 pde loss")

    plts2[0].set_title("at Y=0.5, X")
    plts2[0].scatter(X[128], u_pred[128], s=s, marker='.', label = "Predicted u")
    plts2[0].scatter(X[128], U[128], s=s, marker='.', label = "Exact u")
    plts2[0].legend()

    plts2[1].set_title("Forward Passes vs iterations")
    plts2[1].scatter(info[:,6], info[:,1], marker='.')
    plts2[1].set_yscale("log")
    plts2[1].set_xlabel("Forward passes")
    plts2[1].set_ylabel("log10 L2 relative error")


    plts2[2].set_title("log(10) L2 relative error (PINN)")
    plts2[2].scatter(info[:,0], np.log10(info[:,1]), marker='.')
    plts2[2].set_xlabel("iterations")
    plts2[2].set_ylabel("log10 L2 relative error")

    plts2[3].set_title("Iternation and log(10) total loss ")
    plts2[3].scatter(info[:,0], np.log10(info[:,2]), marker='.')
    plts2[3].set_xlabel("Iterations")
    plts2[3].set_ylabel("log10 composite loss")

    plts2[4].set_title("Iternation and log(10) PINN loss ")
    plts2[4].scatter(info[:,0], np.log10(info[:,3]), marker='.')
    plts2[4].set_xlabel("iterations")
    plts2[4].set_ylabel("log10 PINN loss")

    plts2[5].set_title("Iternation and log(10) PINN PDE loss ")
    plts2[5].scatter(info[:,0], np.log10(np.abs(info[:,4])), marker='.')
    plts2[5].set_xlabel("iterations")
    plts2[5].set_ylabel("log10 PDE loss")

    fig.tight_layout()
    
    
def performancePlot(ax, info, modelType, label = ""):
    ax[0][0].scatter(np.log10(info[:,0]), np.log10(info[:,1]), s=[6 for i in info[:, 0]], label = label)
    
    ax[1][0].scatter(info[:,0], np.log10(info[:,1]), s=[6 for i in info[:, 0]], label = label)
    
    ax[0][0].set_xlabel("log10 training iterations")
    ax[0][0].set_ylabel("log10 relative L2 error")
    ax[0][1].set_xlabel("log10 training iterations")
    ax[0][1].set_ylabel("log10 PINN loss")
    ax[0][2].set_xlabel("log10 training iterations")
    ax[0][2].set_ylabel("log10 PDE loss")
    
    ax[1][0].set_xlabel("training iterations")
    ax[1][0].set_ylabel("log10 relative L2 error")
    ax[1][1].set_xlabel("training iterations")
    ax[1][1].set_ylabel("log10 PINN loss")
    ax[1][2].set_xlabel("training iterations")
    ax[1][2].set_ylabel("log10 PDE loss")
    
#     plts1[0].legend(loc = 4, markerscale=4)

#     plts1[1].set_title("Log 10 Iternation and log(10) PINN loss")
    if label == "Adam" or label == "SGD":
        ax[0][1].scatter(np.log10(info[:,0]), np.log10(info[:,2]), s=[6 for i in info[:, 0]], label = label)
    
        ax[0][2].scatter(np.log10(info[:,0]), np.log10(info[:,3]), s=[6 for i in info[:, 0]], label = label)
        
        ax[1][1].scatter(info[:,0], np.log10(info[:,2]), s=[6 for i in info[:, 0]], label = label)
        
        ax[1][2].scatter(info[:,0], np.log10(info[:,3]), s=[6 for i in info[:, 0]], label = label)
        
    else:
        ax[0][1].scatter(np.log10(info[:,0]), np.log10(info[:,3]), s=[6 for i in info[:, 0]], label = label)
        ax[0][2].scatter(np.log10(info[:,0]), np.log10(info[:,4]), s=[6 for i in info[:, 0]], label = label)
        
        ax[1][1].scatter(info[:,0], np.log10(info[:,3]), s=[6 for i in info[:, 0]], label = label)
        ax[1][2].scatter(info[:,0], np.log10(info[:,4]), s=[6 for i in info[:, 0]], label = label)

    
    for i in ax:
        for j in i:
            j.legend(markerscale = 4)

# plts2[0].set_title("Iternation and log(10) L2 relative error ")

