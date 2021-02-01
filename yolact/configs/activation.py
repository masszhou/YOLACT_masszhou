import torch


activation_func = {
    'tanh': torch.tanh,
    'sigmoid': torch.sigmoid,
    'softmax': lambda x: torch.nn.functional.softmax(x, dim=-1),
    'relu': lambda x: torch.nn.functional.relu(x, inplace=True),
    'none': lambda x: x,
}