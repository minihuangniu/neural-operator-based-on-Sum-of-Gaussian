import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from neuralop.data.datasets.burgers import load_mini_burgers_1dtime
import torch.nn.functional as F
from neuralop import LpLoss, H1Loss
from neuralop.utils import count_model_params

class SOG_block1d(nn.Module):
    def __init__(self,res):
        super().__init__()
        self.res=res
        self.fc0 = nn.Linear(res,128)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, res)
        self.scale = nn.Parameter(torch.rand(256,dtype=torch.float32))

    def forward(self, x):
        B, C, R = x.shape
        x = torch.square(x)
        x = x.view(B, R)
        x = F.tanh(self.fc0(x))
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        x = torch.exp(x)
        x = x * self.scale
        x = self.fc_out(x)
        x = x.view(B, 1, R)
        return x

class SOG1d(nn.Module):
    def __init__(self,resolution=16):
        super().__init__()
        self.res=resolution
        self.sog0 = SOG_block1d(self.res)
        self.sog1 = SOG_block1d(self.res)
        self.sog2 = SOG_block1d(self.res)
        self.sog3 = SOG_block1d(self.res)

        self.w0 = nn.Conv1d(1, 1, 1)
        self.w1 = nn.Conv1d(1, 1, 1)
        self.w2 = nn.Conv1d(1, 1, 1)
        self.w3 = nn.Conv1d(1, 1, 1)

    def forward(self,x):
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
def burgers_test(model, test_loaders, time_u, criterion, device, resolution=16):
    model.eval()
    test_loss = 0.0
    test_samples = 0
    with torch.no_grad():
        for data in test_loaders[resolution]:
            a = data['x'].to(device)
            u = data['y'][:,:,time_u,:].to(device)
            test_bt = a.size(0)

            output = model(a)
            loss = criterion(output, u)

            test_loss += (loss.item() * test_bt)
            test_samples += test_bt

    return test_loss / test_samples

def plot_burgers(model, time_u, device, test_loaders, resolution=16):
    model.eval()
    with torch.no_grad():
        for data in test_loaders[resolution]:
            a = data['x'].to(device)
            u = data['y'][:, :, time_u, :].to(device)
            y_pred = model(a)
            true_u = u[0, 0].cpu().numpy()
            pred_u = y_pred[0, 0].cpu().numpy()

            x = np.linspace(0, 1, len(true_u))

            plt.figure(figsize=(8, 8))
            plt.plot(x, true_u, label='Ground Truth', marker='o')
            plt.plot(x, pred_u, label='Prediction', marker='x')
            plt.xlabel('x')
            plt.ylabel(f'u(x, t={time_u})')
            plt.title(f'Burgers Equation Fitting (t={time_u})')
            plt.legend()
            plt.grid(True)
            plt.show()
            break

if '__main__' == __name__:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    epochs = 1500
    learning_rate = 2e-3
    weight_decay = 1e-5
    n_train = 800
    n_test = 400
    batchsize = 32
    test_batchsize = 32
    stepsize = 100
    gamma = 0.5
    time_u = 0

    DATA_DIR = Path('../neuralop/data/datasets/data')



    train_loader, test_loaders, data_processor = load_mini_burgers_1dtime(
        DATA_DIR,
        n_train=n_train,
        n_test=n_test,
        batch_size=batchsize,
        test_batch_size=test_batchsize
    )
    print(test_loaders.keys())
    t = 0
    for data in test_loaders[16]:
        print(data['x'].size(0))
        t += data['x'].size(0)
    print(t)



    model = SOG1d()
    model.to(device)
    n_params = count_model_params(model)
    print(f'\nThe model has {n_params} parameters.')
    criterion = LpLoss(d=1, p=2, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=gamma)

    training_loss = []
    for epoch in range(1,epochs+1):
        model.train()
        total_loss = 0
        total_samples = 0
        for i, data in enumerate(train_loader, 0):
            a = data['x'].to(device)
            u = data['y'][:,:,time_u,:].to(device)
            batch_size = a.size(0)
            optimizer.zero_grad()
            y_pred = model(a)
            loss = criterion(y_pred, u)
            loss.backward()
            optimizer.step()
            total_loss += (loss.item() * batch_size)
            total_samples += batch_size
            training_loss.append(loss.item())
            print(f'Epoch: {epoch}/{epochs} Total Loss: {loss.item():.6f}')
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()


    test_loss = burgers_test(model, test_loaders, time_u, criterion, device)
    print(f'Loss on test set of res=16 is: {test_loss:.6f}')

    plt.figure(figsize=(8, 8))
    plt.plot(training_loss, label='train loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training Loss per Steps')
    plt.legend()
    plt.show()

    plot_burgers(model, time_u, device, test_loaders)






