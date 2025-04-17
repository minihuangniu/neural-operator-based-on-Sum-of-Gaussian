import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import sys
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.data.datasets import load_mini_burgers_1dtime
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


project_root = Path(__file__).parent.parent
data_dir = project_root / "neuralop" / "data" / "datasets" / "data"
train_file = data_dir / "burgers_train_16.pt"


#load burgers dataset
data_path = data_dir
n_train = 800
n_test = 0
batch_size = 32
test_batch_size = 32

train_loader, test_loaders, data_processor = load_mini_burgers_1dtime(
    data_path,
    n_train=n_train,
    n_test=n_test,
    batch_size=batch_size,
    test_batch_size=test_batch_size
)
data_processor = data_processor.to(device)
for data in train_loader:
    print(type(data['x']))
    print(data['x'].shape)
    print(type(data['y']))
    print(data['y'].shape)
    break


# Create a simple FNO model
model = FNO(n_modes=(16,),
             in_channels=1,
             out_channels=1,
            hidden_channels=24,
            lifting_channel_ratio=1,
            projection_channel_ratio=1)
model = model.to(device)

n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()

# Training setup
#Create the optimizer
optimizer = AdamW(model.parameters(),
                                lr=8e-3,
                                weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# Then create the losses
l2loss = LpLoss(d=1, p=2)
h1loss = H1Loss(d=1)

train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}

# Training the model
# ---------------------

print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()

# Create the trainer:
trainer = Trainer(model=model, n_epochs=200,
                  device=device,
                  wandb_log=False,
                  eval_interval=20,
                  use_distributed=False,
                  verbose=True)

# Then train the model

trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer,
              scheduler=scheduler,
              regularizer=False,
              training_loss=train_loss,
              eval_losses=eval_losses)


