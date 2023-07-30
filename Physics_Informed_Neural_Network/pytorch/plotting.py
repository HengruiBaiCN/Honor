import numpy as np
import torch.cuda

from pyDOE import lhs
from torch import from_numpy

import matplotlib.pyplot as plt


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