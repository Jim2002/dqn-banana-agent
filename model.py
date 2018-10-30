import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        layers_size = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(n1, n2) for n1, n2 in layers_size])
        self.output = nn.Linear(hidden_layers[-1], action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        count = 0
        
        for linear in self.hidden_layers:
            if count > 0:
                x = F.relu(linear(x))
            else:
                x = F.relu(linear(state))
                count += 1 
                
        
        return self.output(x)
