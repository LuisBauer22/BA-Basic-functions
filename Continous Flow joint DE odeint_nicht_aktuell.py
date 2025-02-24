# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:25:57 2025

@author: luis-
"""

import sympy as smp
import numpy as np
from scipy.stats import cauchy
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import approx_fprime
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import scipy.stats
import time


start_time =time.time()

#Dim, first do 1-dim case
D=2

#parameter vector Theta
Θ = [5, 2]

#here you could use any differential equation
# def u(t, X):
#     "vector field "
#     "solves for x1, x2 = X "
#     x1, x2, x3  =X

#     #dividing by t makes problems
#     #non-linear dependancies of x1, x2 make problems
#     return [-2*x1**1 + 0.1*x2**1 - 0.5*x3, 
#             -3*x1**1 +   5*x2**1 + 0.2*x3,
#             -1*x1    +               7*x3]

#here you could use any differential equation
def u(t, X):
    "vector field "
    "solves for x1, x2 = X "
    x1, x2,  =X

    #dividing by t makes problems
    #non-linear dependancies of x1, x2 make problems
    return [-2*x1**1 + 0.1*x2**1, 
            -3*x1**1 +   1.5*x2**1]
    

def dxdt(t, X):
    """solves for dx/dt, return array(x1, x2)
       d/dt X = u(t, x)"""
    return u(t, X)


def div_u_num_jac(t, X):
    """ Computes the divergence using Scipy's approx_fprime 
        via computing the whole jacobian=>O(D^2) instead O(D)
        =>extremly inefficient for higher dimensions"""
    epsilon = np.sqrt(np.finfo(float).eps)  # Optimal step size
    jacobian= approx_fprime(X, lambda X: u(t, X), epsilon)
    
    return np.trace(jacobian)


def div_u_num_fast(t, X):
    """ Computes the divergence of u(t, X) efficiently using finite differences.
        Only computes the necessary diagonal elements of the Jacobian.
    """
    epsilon = np.sqrt(np.finfo(float).eps)  # Optimal step size??
    D = len(X)  # Dimension of X
    divergence = 0  # Initialize divergence sum

    # Compute only diagonal entries (∂uᵢ/∂Xᵢ) using one-sided finite differences
    for i in range(D):
        X_forward = np.array(X)
        X_forward[i] += epsilon  # Perturb only the i-th coordinate
        divergence += (u(t, X_forward)[i] - u(t, X)[i]) / epsilon  # Approximate derivative

    return divergence
    #This is 4x faster than the jacobian


#we would need to evaluate u over d-dim grid first and then calculate div
#by iterating over(np.grad(u, epsilon, axis=i))


def d_log_pt_dt(t, X):
    """d/dt log pt(Xt) = -(∇ ut)(Xt)
       takes in time and X-vector, 
       returns scalar"""
    
    #X_t is 2D here
    return - div_u_num_fast(t, X)

def joint_ODE(t, S):
    "solves for X and log_pt at the same time"
    "input vector S = x1, x2, log_pt(x1, x2);  "
    "return vector of x1, x2, log_pt(X)"
    log_pt = S[-1] # is the last component of S
    X = S[:-1]
    joint_sol = np.append(u(t, X), d_log_pt_dt(t, X))
    return joint_sol


num_steps = 100
#could depend on error margin, 
#we could also use adaptive time steps later

t=np.linspace(0, 1, num_steps)
#t goes from 0 to 1, 

#drawinit_init_samplesfrom basic gaussian in a reproducible way
np.random.seed(10) 
num_samples=1000
mu= [0, 1, 2]
cov_matr =  [[2, 0, 0], 
             [0, 2, 0], 
             [0, 0, 2]]

mu= [0, 1]
cov_matr =  [[2, 0], 
             [0, 2]]

p0   = multivariate_normal(mu, cov_matr)
init_samples= np.random.multivariate_normal(mu, cov_matr, size=num_samples)
pdf_init_samples= p0.pdf(init_samples) 
log_pdf_init_samples= np.log(pdf_init_samples)

tot_init_samples= np.column_stack((init_samples, log_pdf_init_samples))#array for initial cond. x1, x2 ... xD, log_pt

#S=(x1, x2, ..., xD,  log_pt(X))

sol_m1 = np.empty([ num_samples, num_steps, D+1])
                    #100,         100         4 

j=0
#Now use one loop for all 3 variables x1, x2, log pt(x1, x2)
#solve with joint ODE
for s0 in tot_init_samples:
    sol_m1[j]=odeint(joint_ODE, y0=s0, t=t, tfirst=True)#, full_output=1)
    #saves the solution [x1 ,x2 ..., log_pt] for a specific sample initial condition s0
    j+=1


#use 2 nested loops to plot all the dimensions of the evolution for all theinit_samples

#iterate over the different dimensions of the solution    
for i in range(D): #master loop for different dimensions, #3
    for j in range(num_samples): #loop overinit_init_samples
        plt.plot(t, sol_m1[j].T[i])
    str_xi_t = "x" + str(i+1) +"(t)"
    plt.title(str_xi_t + " with joint log ODE")
    plt.ylabel(str_xi_t, fontsize=18)
    plt.xlabel(str_xi_t, fontsize=18)
    plt.show()
    plt.clf()
    

#loop for plotting log pt

#last entry (D) is log_pt
#log_pt_sol_m1[j] = sol_m1[j].T[D]
for j in range(num_samples):
    plt.plot(t, sol_m1[j].T[D])

plt.title("log_pt(t) with log joint solver odeint")
plt.ylabel('$log [p_t(t)]$', fontsize=18)
plt.xlabel('$t$', fontsize=18)
plt.show()
plt.clf() 

end_time = time.time()   

print("end time = " + str(end_time - start_time))

#%%
#save the end results of each ODE as an array
transformed_samples=np.zeros((D+1, num_samples))


#save end point for allinit_init_samplesas array(2D, dim-samples, dim-x1, x2...log_pt)
for i in range(D+1):
    for j in range(num_samples):
        #num_steps_j = len(sol_m1[j].T)-1
        transformed_samples[i, j] = sol_m1[j].T[i][num_steps-1]


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,8))
fig.suptitle("2D-Histograms and log densities", fontsize=22) 

num_bins=50

x =init_samples.T[0]
y =init_samples.T[1]
hist_init=axs[0][0].hist2d(x, y, bins=(num_bins, num_bins), cmap=plt.cm.jet)
axs[0][0].set_title('Initial_samples')
fig.colorbar(hist_init[3], ax=axs[0, 0])  # Add colorbar


x_T = np.array(transformed_samples[0])
y_T = np.array(transformed_samples[1])
hist_tf=axs[0][1].hist2d(x_T, y_T, bins=(num_bins, num_bins), cmap=plt.cm.jet)
axs[0][1].set_title('Transformed samples')
fig.colorbar(hist_tf[3], ax=axs[0, 1])  # Add colorbar

N = 51
# Create Meshgrid for initial density
x_min, x_max = min(init_samples.T[0]), max(init_samples.T[0])
y_min, y_max= min(init_samples.T[1]), max(init_samples.T[1])
x = np.linspace(x_min,x_max, N)
y = np.linspace(y_min,y_max, N)
#z = np.linspace(0, 0, 1)
xx, yy = np.meshgrid(x, y)
points = np.stack((xx, yy), axis=-1)
pdf = np.log(scipy.stats.multivariate_normal.pdf(points, mu, cov_matr))
init_log_pdf =axs[1][0].imshow(pdf, extent=[x_min,x_max,y_min,y_max], 
                 origin="lower",  cmap="jet")
axs[1][0].set_title('Initial log density')
fig.colorbar(init_log_pdf, ax=axs[1, 0])

#In 3d müssen wir Transformed_samples[3] machen
#Muss hier irgendwie log_pt am Ende plotten
#log_pt sind skalare, die die log dichte am Punkt p_t am Ende beschreiben
#z= log_pt (x1, x2 ), histogramm-mäßig
z = transformed_samples[D]
tf_log_pdf=axs[1][1].scatter(x_T, y_T, c=z, cmap='jet',
                  marker='o', s=0.1)
axs[1][1].set_title('Transformed samples log pdf')
fig.colorbar(tf_log_pdf, ax=axs[1, 1])
plt.show()
plt.clf()

