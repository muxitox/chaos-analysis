#!/usr/bin/env python

import numpy as np
import autograd.numpy as anp
from matplotlib import pyplot as plt
from autograd import grad, jacobian

plt.rc('text', usetex=True)
font = {'size':15}
plt.rc('font',**font)
plt.rc('legend',**{'fontsize':12})


def sigma(x):
    return 0.5 * (1+anp.tanh(x/2))
#    return 1/(1+np.exp(-x))
    

def step(y_0):

    #    These three equations are equivalent by applying a change of variables
    y = y_0 + dt / tau * (-y_0 + 0.5 * anp.tanh(0.5 * (W.T @ (y_0 + 0.5) + th)))
    return y

N=3

mode = "periodic"
tau = np.ones(N)
if mode == "periodic":
    W = np.ones((N,N))
    di = np.diag_indices(N)
    W[di] = 10
    th = np.ones(N) * -6
else:
    # saddle_limit
    W = np.array([[5.422, -0.24, 0.535], [-0.018, 4.59, -2.25], [2.75, 1.21, 3.885]])
    th = np.array([-4.108, -2.787, -1.114])
    if mode == "periodic_limit":
        tau[1] = 1.92
    elif mode == "chaotic":
        tau[1] = 2.5


y_0=np.random.randn(N)*0.1

dt = 0.01

Jacobian_Func = jacobian(step)

dx = np.identity(N)
S = np.zeros(N)

transient_steps = 400000

running_steps = 400000
T = transient_steps + running_steps

y=np.zeros((N, running_steps))
# Debugging variable
S_i = np.zeros((N, running_steps))


for t in range(T):
    y_0 = step(y_0)

    if t  >= transient_steps:
        y[:, t - transient_steps] = y_0

        J = Jacobian_Func(y_0)

        # perturbation
        dx = np.matmul(J, dx)

        Q, R = np.linalg.qr(dx)
        d_exp = np.absolute(np.diag(R))
        dS = np.log(d_exp)

        # Q is orthogonal so we can use it as the perturbation for the next step
        dx = Q

        S += dS

        S_i[:, t-transient_steps] = S

S /= running_steps

plt.figure()
plt.plot(y.T)
plt.title("Trajectories")

ax = plt.figure().add_subplot(projection='3d')
ax.plot(y[0,:], y[1,:], y[2,:])
plt.show()


print("Lyapunov exponents", S)

plt.figure()
plt.plot(S_i[0,:])
plt.title(fr"$\hat{{\lambda_0}}/M = {S[0]}$")
plt.tight_layout()
plt.figure()
plt.plot(S_i[1,:])
plt.title(fr"$\hat{{\lambda_1}}/M = {S[1]}$")
plt.tight_layout()
plt.figure()
plt.plot(S_i[2,:])
plt.title(fr"$\hat{{\lambda_2}}/M = {S[2]}$")
plt.tight_layout()
plt.show()



