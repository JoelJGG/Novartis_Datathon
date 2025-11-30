
import torch
import torch.nn as nn

class ModelNovartis(nn.Module): 
    def __init__(self,inputs=97,outputs=24):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(inputs,180),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(180,144),
            nn.LeakyReLU(),
            nn.Linear(144,77),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(77,outputs),
            nn.Softplus()
        )

    def forward(self,X):
        return self.features(X)

