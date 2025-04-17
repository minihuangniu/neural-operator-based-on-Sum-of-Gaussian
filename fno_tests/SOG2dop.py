import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from neuralop.utils import get_project_root
from neuralop import LpLoss, H1Loss
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params

class SOG_block2d(nn.Module):
    def __init__(self, in_channels, resolution):
        super().__init__()
        self.res = resolution
        self.flatten_size = in_channels * resolution * resolution

        self.fc0 = nn.Linear(self.flatten_size, 512)
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc_out = nn.Linear(1024, resolution * resolution)
        self.scale = nn.Parameter(torch.rand(1024, dtype=torch.float32))

    def forward(self, x):
        B, C, H, W = x.shape
        x = torch.square(x)
        x = x.view(B, -1)  # Flatten

        x = F.tanh(self.fc0(x))
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        x = torch.exp(x) * self.scale
        x = self.fc_out(x)
        x = x.view(B, 1, H, W)  # 输出是单通道
        return x


class SOG2d(nn.Module):
    def __init__(self, in_channels=1, resolution=16):
        super().__init__()
        self.res = resolution
        self.in_channels = in_channels

        self.sog0 = SOG_block2d(in_channels + 2, resolution)
        self.sog1 = SOG_block2d(1, resolution)
        self.sog2 = SOG_block2d(1, resolution)
        self.sog3 = SOG_block2d(1, resolution)

        self.w0 = nn.Conv2d(in_channels + 2, 1, 1)
        self.w1 = nn.Conv2d(1, 1, 1)
        self.w2 = nn.Conv2d(1, 1, 1)
        self.w3 = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        # 输入形状：[B, 1, H, W]
        B, C, H, W = x.shape

        # 生成归一化网格
        grid_x = torch.linspace(0, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        grid_y = torch.linspace(0, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        grid = torch.cat([grid_x, grid_y], dim=1)

        x = torch.cat([x, grid], dim=1)  # x: [B, C+2, H, W]

        x1 = self.sog0(x)
        x2 = self.w0(x)
        x = x1 + x2

        x1 = self.sog1(x)
        x2 = self.w1(x)
        x = x1 + x2

        x1 = self.sog2(x)
        x2 = self.w2(x)
        x = x1 + x2

        x1 = self.sog3(x)
        x2 = self.w3(x)
        x = x1 + x2

        return x


def darcy_test(model, test_loaders, resolution, criterion, device):
    # 获取对应分辨率的测试集
    test_loader = test_loaders.get(resolution)
    if test_loader is None:
        print(f"没有找到分辨率 {resolution} 的测试集！")
        return

    model.eval()
    test_loss = 0.0
    test_samples = 0
    with torch.no_grad():
        for data in test_loader:
            a = data['x'].to(device)
            u = data['y'].to(device)
            testbt = a.size(0)
            output = model(a)
            loss = criterion(output, u)

            test_loss += (loss.item() * testbt)
            test_samples += testbt

    return test_loss / test_samples


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    epochs = 1000
    learning_rate = 6e-4
    weight_decay = 1e-5
    n_train = 1000
    n_tests = [50, 50]
    test_resolutions = [16, 32]
    batchsize = 32
    test_batchsize = [32, 32]
    stepsize = 100
    gamma = 0.5

    DATA_DIR = Path('../neuralop/data/datasets/data')

    train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=n_train,
        batch_size=batchsize,
        test_resolutions=test_resolutions,
        n_tests=n_tests,
        test_batch_sizes=test_batchsize,
    )


    model = SOG2d()
    model.to(device)
    n_params = count_model_params(model)
    print(f'\nThe model has {n_params} parameters.')
    criterion = LpLoss(d=2, p=2, reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    training_loss = []
    for epoch in range(1,epochs+1):
        model.train()
        total_loss = 0
        total_samples = 0
        for data in train_loader:
            a = data['x'].to(device)
            u = data['y'].to(device)
            batch_size = a.size(0)
            optimizer.zero_grad()
            y_pred = model(a)
            loss = criterion(y_pred, u)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            training_loss.append(loss.item())
    
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        print(f'Epoch: {epoch}/{epochs} Total Loss: {total_loss / total_samples:.6f} ')

    plt.figure(figsize=(8, 8))
    plt.plot(training_loss, label='train loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training Loss per Steps')
    plt.legend()
    plt.show()


    test_loss_16 = darcy_test(model, test_loaders, resolution=16, criterion=criterion, device=device)
    #test_loss_32 = darcy_test(model, test_loaders, resolution=32, criterion=criterion, device=device)
    print(f'Testing loss on resolution 16: {test_loss_16}')
    #print(f'Testing loss on resolution 32:{test_loss_32}')


