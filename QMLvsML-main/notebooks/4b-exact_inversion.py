import numpy as np
import torch
from torch import nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split 
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MeanSquaredError
from itertools import product


def fourier_best_approx(W, x, y):
    W = W.t()
    z = x.matmul(W)
    A_cos = z.cos()
    A_sin = z.sin()
    A = torch.cat((A_cos, A_sin), 1)
    coeffs = torch.linalg.lstsq(A, y, driver='gelsd').solution.squeeze(-1)
    return coeffs

def freq_generator(max_freq, dim, mode="positive"):
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


mse = MeanSquaredError()

def loss(W, C, X, y):
    z = X.matmul(W.t())
    A_cos = z.cos()
    A_sin = z.sin()
    A = torch.cat((A_cos, A_sin), 1)
    y_pred = A.matmul(C.t())
    y_pred = y_pred.squeeze()  # Adjust the shape of y_pred
    return mse(y_pred, y.squeeze())  # Ensure y is also squeezed

# Data pre processing

# Fetch data
data = fetch_california_housing()
X = torch.from_numpy(data.data).float()
y = torch.from_numpy(data.target).float()

# Split dataset into training + validation and testing
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Further split into separate training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=2)

# Calculate scaling parameters from the training set
min_value, max_value = X_train.min(dim=0)[0], X_train.max(dim=0)[0]

# Scale the datasets
X_train_scaled = data_scaler(X_train, min_value=min_value, max_value=max_value, interval=(-torch.pi/2, torch.pi/2))
X_val_scaled = data_scaler(X_val, min_value=min_value, max_value=max_value, interval=(-torch.pi/2, torch.pi/2))
X_test_scaled = data_scaler(X_test, min_value=min_value, max_value=max_value, interval=(-torch.pi/2, torch.pi/2))

# Targets do not need scaling for regression tasks but ensure they are tensors
y_train_tensor = y_train.unsqueeze(1)
y_val_tensor = y_val.unsqueeze(1)
y_test_tensor = y_test.unsqueeze(1)

# Generate frequencies
max_freq = 2  
dim = X_train.shape[1]
W = freq_generator(max_freq, dim, mode="half").float()

# Computing the coefficients
coeffs2 = fourier_best_approx(W, X_train_scaled, y_train_tensor )
print(coeffs2)

# Computing the losses
# Calculate losses
train_loss2 = loss(W, coeffs2, X_train_scaled, y_train_tensor)
val_loss2 = loss(W, coeffs2, X_val_scaled, y_val_tensor)
test_loss2 = loss(W, coeffs2, X_test_scaled, y_test_tensor)

print(f"Train Loss: {train_loss2}")
print(f"Validation Loss: {val_loss2}")
print(f"Test Loss: {test_loss2}")
