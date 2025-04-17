import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class SOG(nn.Module):
    def __init__(self):
        super(SOG, self).__init__()
        self.dense1 = nn.Linear(in_features=1, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=256)
        self.dense3 = nn.Linear(in_features=256, out_features=128)
        self.scale = nn.Parameter(torch.empty(128, dtype=torch.float32,device='cuda:0'))
        nn.init.normal_(self.scale)

    def forward(self, x):
        x2 = torch.square(x)
        dense1 = F.tanh(self.dense1(x2))
        dense2 = F.tanh(self.dense2(dense1))
        dense3 = self.dense3(dense2)
        exp_activation = torch.exp(dense3)
        scaled_output = exp_activation * self.scale
        summed = torch.sum(scaled_output, dim=-1, keepdim=True)
        return summed


class pointsdataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class RelativeErrorLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(RelativeErrorLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        diff = y_true - y_pred
        relative_error = torch.norm(diff, p=2, dim=0, keepdim=True) / (
                torch.norm(y_true, p=2, dim=0, keepdim=True) + self.epsilon)
        return relative_error


'''
------------------------------------------------------------------------
'''

learning_rate = 1e-3
weight_decay = 1e-4
epochs = 3000
batchsize = 1000

x = torch.linspace(1e-6, 6, 10000, dtype=torch.float32).reshape(-1, 1)
print(x)
y_true = torch.cos(x).reshape(-1, 1)
dataset = pointsdataset(x, y_true)
dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

model = SOG()
model.to(device)
criterion = RelativeErrorLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.96)

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    for data in dataloader:
        x, y = data
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)

        optimizer.zero_grad()
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    current_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {avg_loss:.6f}, learning_rate: {current_lr:.6f}")

'''
class PhaseShift(nn.Module):
    def __init__(self,):
        super(PhaseShift, self).__init__()
        self.dense1 = nn.Linear(1,16)
        self.dense2 = nn.Linear(16,32)
        self.dense3 = nn.Linear()

    def forward(self, inputs):
        phi_x = torch.tanh(self.dense1(inputs))
        phi_x = torch.tanh(self.dense2(phi_x))
        phi_x = self.dense3(phi_x) * torch.pi  # 控制相位大小
        return inputs + phi_x  # 进行相位偏移
'''
