import numpy as np 
from itertools import product
import torch

def freq_generator(max_freq, dim, mode="positive"):
    """
    Generates the frequencies for the multi-dim partial Fourier series. 
    Returns a tensor of the form 
    ((-max_freq, ..., -max_freq), (-max_freq, ..., -max_freq+1), ..., (max_freq, ..., max_freq))
    In other words: The function generates the nodes of the n-dim integer lattice Z^n but Z is 
    limited to abs(integers) smaller equal max_freq.
    note. The number of frequencies created is (max_freq + 1)**dim

    Parameters:
    max_freq (int) : highest integer available for the lattice.
    dim (int) : dimension of the lattice respective the frequencies
    """
    if mode == "positive":
        w = list(np.arange(0, max_freq+1, 1)) # all the integers to build \N^n
        W = list(product(w, repeat=dim))
    elif mode == "all":
        w = list(np.arange(-max_freq, max_freq+1, 1)) # all the integers to build \Z^n
        W = list(product(w, repeat=dim))
    elif mode == "half":
        W = []
        Pos = list(np.arange(1, max_freq+1, 1)) 
        All = list(np.arange(-max_freq, max_freq+1, 1))
        for d in range(dim):
            w = list(np.zeros(d, dtype=np.intc)) # the zero coordinates
            for j in range(len(Pos)):
                All_combinations = list(product(All, repeat=dim-d-1))
                for k in range(len(All_combinations)):
                    h = w + [Pos[j]]+list(All_combinations[k])
                    W.append(h)
        W.append(list(np.zeros(dim, dtype=np.intc)))
    return torch.tensor(W, dtype=torch.float64)




def data_scaler(x, min_value=None, max_value=None, standard_deviation=None, interval=None):
    """
    Scales data to either 
    a) (interval = None) mean 0 and a given standard deviation
    b) such that all data points lie in a specified interval using provided min and max values.

    Parameters:
        x (torch.tensor): raw data to be rescaled.
        min_value (torch.tensor): minimum value used for scaling, should be calculated from the training set.
        max_value (torch.tensor): maximum value used for scaling, should be calculated from the training set.
        standard_deviation (bool or float): standard_deviation of the rescaled data, default is None which 
            corresponds to standard_deviation = 1.
        interval (bool or tuple): target interval of the rescaling, given as (lower bound, upper_bound).
    
    Returns:
        x (torch.tensor): Rescaled data.
    """
    # Check for correct input
    assert (min_value is not None and max_value is not None) or interval is None, \
        "Provide min_value and max_value for scaling to an interval, or set interval to None for standardization."

    if interval is not None:
        # Rescale to the interval using provided min and max
        x = (x - min_value) / (max_value - min_value) * (interval[1] - interval[0]) + interval[0]

    else:
        # if no interval is given, set standard_deviation to default (standard_deviation=1)
        if standard_deviation is None:
            standard_deviation = 1

        if standard_deviation is not None:
            m = x.mean(0, keepdim=True)
            s = x.std(0, unbiased=False, keepdim=True)
            x -= m
            x /= s
            x *= standard_deviation

    return x



def data_scaler2(x, standard_deviation=None, interval=None):
    """
    Scales data to either 
    a) (interval = None) mean 0 and a given standard deviation
    b) such that all data points lie in a specified interval

    Parameters:
        x (torch.tensor) : raw data to be rescaled.
        standard_deviation (bool or float) : standard_deviation of the rescaled data, default is None which 
            corresponds to standard_deviation = 1.
        interval (bool or tuple) : target interval of the rescaling, given as (lower bound, upper_bound).
    
    Returns:
        x (torch.tensor) : Rescaled data.
    
    """
    if interval is not None:
        # check for correct input
        assert standard_deviation is None, "interval=None and standard_deviation=None is not possible at the same time!"

        # rescale
        x = (x - x.min()) / (x.max() - x.min()) * (interval[1] - interval[0]) + interval[0]

    # if no interval is given, set standard_deviation to default (standard_deviation=1)
    if interval is None and standard_deviation is None:
        standard_deviation = 1

    if standard_deviation is not None:
        m = x.mean(0, keepdim=True)
        s = x.std(0, unbiased=False, keepdim=True)
        x -= m
        x /= s
        x *= standard_deviation

    return x 


def loss(W, C, X, y):
    z = X.matmul(W.t())
    A_cos = z.cos()
    A_sin = z.sin()
    A = torch.cat((A_cos, A_sin), 1)
    y_pred = A.matmul(C.t())
    y_pred = y_pred.squeeze()  # Adjust the shape of y_pred
    return mse(y_pred, y.squeeze())  # Ensure y is also squeezed


def fourier_best_approx(W, x, y):
    """
    Given frequencies w of a fourier type series, computes the coefficients such that
    the euclidean norm of f(x_i)-y_i is minimized.

    W (torch.tensor) : gives a tensor of n frequencies, each of size (1, l), all in all
        a tensor of size (n, l)
    
    x (torch.tensor) : the m datapoints corresponding to the labels y_i, each datapoint x_i is of
        dimension (1, l), such that x is (m, l)
    
    y (torch.tensor) : the m labels, dimension (1,m)
    """
    W = W.t()
    z = x.matmul(W)
    A_cos = z.cos()
    A_sin = z.sin()
    A = torch.cat((A_cos, A_sin), 1)
    coeffs = torch.linalg.lstsq(A,y, driver='gelsd').solution
    return coeffs


