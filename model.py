
import torch
import torch.nn as nn

class ModelNovartis(nn.Module): 
    def __init__(self,inputs=72,outputs=24):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(inputs,100),
            nn.ReLU(),
            nn.Linear(100,144),
            nn.ReLU(),
            nn.Linear(144,288),
            nn.ReLU(),
            nn.Linear(288,188),
            nn.ReLU(),
            nn.Linear(188,144),
            nn.ReLU(),
            nn.Linear(144,96),
            nn.ReLU(),
            nn.Linear(96,48),
            nn.ReLU(),
            nn.Linear(48,outputs),
            nn.Softplus()
        )

    def forward(self,X):
        return self.features(X)

