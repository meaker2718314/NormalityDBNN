import torch.nn as nn

""" 
Model architecture inspired by Miloš Simić,
(arXiv:2009.13831, published in Neural Computing and Applications, 2021).
 """


class DBNN(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
