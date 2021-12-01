import torch
import pandas as pd
import matplotlib.pyplot
import numpy as np

from utils.black_scholes import black_scholes_put

# domain boundaries
S_bound = [0.0, 200.0]
K_bound = [50.0, 150.0]
T_bound = [0.0, 5.0]
r_bound = [0.001, 0.05]
sigma_bound = [0.05, 1.5]

def generate_unif(n):
    # insert your smarter strategy here
    return (np.random.rand(n,5))

def generate_black_scholes_put_data(n):

    X = generate_unif(n)

    S_delta = S_bound[-1] - S_bound[0]
    K_delta = K_bound[-1] - K_bound[0]
    T_delta = T_bound[-1] - T_bound[0]
    r_delta = r_bound[-1] - r_bound[0]
    sigma_delta = sigma_bound[-1] - sigma_bound[0]

    deltas = np.array([S_delta, K_delta, T_delta, r_delta, sigma_delta])
    l_bounds = np.array([S_bound[0], K_bound[0], T_bound[0], r_bound[0], sigma_bound[0]])

    X = X * deltas + l_bounds
    y = black_scholes_put(S = X[:,0], K = X[:,1], T = X[:,2], r = X[:,3], sigma = X[:,4]).reshape(-1,1)

    return (np.append(X,y, axis = 1))

if __name__ == "__main__":

    XY = generate_black_scholes_put_data(1000)
    XY_df = pd.DataFrame(XY, columns=["S","K","T","r","sigma","value"])

    XY_df.to_csv("bs-put-1k.csv")
