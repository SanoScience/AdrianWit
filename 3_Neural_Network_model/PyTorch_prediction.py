import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#######################################################################################################
# Model definitions by PyTorch
#######################################################################################################


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(6, 128)   # 6 wejść (3 pozycje + 3 siły), 128 neuronów w warstwie
        self.fc2 = nn.Linear(128, 64)  # 64 neurony w drugiej warstwie
        self.fc3 = nn.Linear(64, 3)    # 3 wyjścia (3 pozycje)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # ReLU
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
#######################################################################################################
# Prediction
#######################################################################################################

# Wczytywanie model:
net.load_state_dict(torch.load('network_PyTorch.pkl'))
# Ustawienie modelu w tryb ewaluacji
net.eval()


