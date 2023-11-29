import torch
import torch.nn as nn
import torch.nn.functional as F

class ReLURegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        """
        Initialize the ReLU-based regression model.

        Parameters:
        input_dim (int): The number of features in the input data.
        hidden_dim (int): The size of the hidden layers. This is a hyperparameter that you can tune.
                          Increasing the size can increase the model's capacity, but also the risk of overfitting.
        output_dim (int): The size of the output layer. For regression, this is typically 1 as we predict a single value.
        """
        super().__init__()
        # First fully connected layer with 'input_dim' input features and 'hidden_dim' output features.
        self.fc1 = nn.Linear(input_dim, hidden_dim).double()

        # Second fully connected layer, also with 'hidden_dim' units.
        # Using more than one hidden layer can help the model learn more complex representations,
        # but increases the risk of overfitting.
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).double()

        # Output layer with 'hidden_dim' input features and 'output_dim' output feature.
        self.fc3 = nn.Linear(hidden_dim, output_dim).double()
        
    def forward(self, x):
        """
        Defines the forward pass of the model.

        Parameters:
        x (Tensor): The input data.

        Returns:
        Tensor: The output of the model.
        """
        # Apply ReLU activation function after each linear layer (except the output layer).
        # ReLU helps with the vanishing gradient problem and introduces non-linearity.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # No activation function is applied at the output layer for a regression problem.
        output = self.fc3(x)
        return output

