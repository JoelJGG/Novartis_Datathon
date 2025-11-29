import torch
import torch.nn as nn

class ModelNovartis(nn.Module): 
    def __init__(self,inputs=72,outputs=24):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(inputs,664),
            nn.ReLU(),
            nn.Linear(664,1328),
            nn.ReLU(),
            nn.Linear(1328,864),
            nn.ReLU(),
            nn.Linear(864,432),
            nn.ReLU(),
            nn.Linear(432,outputs)
        )

    def forward(self,X):
        return self.features(X)

