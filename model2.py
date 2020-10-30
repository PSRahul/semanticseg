from pythonfiles.data_class import SemanticDataset
from pythonfiles.nn3 import pl_nn

from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pathlib
import os
from pytorch_lightning import LightningModule

hparams = {"dropout_p": 0.2,
           "batch_size": 8,
           "filter_size_max": 512,
           "weight_decay": 0,
           "max_epochs": 25,
           "save_string": "512p2l20"
           }
model_file_name = "epoch" + \
    str(hparams["max_epochs"])+hparams["save_string"] + \
    str(datetime.now().strftime("%d%m%H%M"))

train_data = SemanticDataset(train=True, val=False, test=False, transform=None)
train_dataloader = DataLoader(
    train_data, batch_size=hparams["batch_size"], num_workers=6)

val_data = SemanticDataset(train=False, val=True, test=False, transform=None)
val_dataloader = DataLoader(
    val_data, batch_size=hparams["batch_size"], num_workers=6)

test_data = SemanticDataset(train=False, val=False, test=True, transform=None)
test_dataloader = DataLoader(
    test_data, batch_size=hparams["batch_size"], num_workers=6)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using the  Device - ", device)
model = pl_nn(hparams=hparams)
model = model.to(device)

# Weight Initialisation


def init_weights(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)


model = model.apply(init_weights)

tb_save_dir = os.path.join(os.getcwd(), 'runs')
cp_save_dir = os.path.join(os.getcwd(), "CKP", model_file_name)


logger = TensorBoardLogger(
    save_dir=tb_save_dir,
    name=model_file_name
)

checkpoint_callback = ModelCheckpoint(
    filepath=cp_save_dir,
    save_top_k=1,
    verbose=True,
    monitor='loss_val',
    mode='min'
)

early_stop_callback = EarlyStopping(monitor='loss_val', verbose=True, mode=min)


trainer = pl.Trainer(gpus=1, max_epochs=hparams["max_epochs"], weights_summary=None,
                     logger=logger, checkpoint_callback=checkpoint_callback, callbacks=[early_stop_callback])

trainer.fit(model, train_dataloader, val_dataloader)

print("Best Model Path", checkpoint_callback.best_model_path)
best_model_path = checkpoint_callback.best_model_path

print(trainer.test(model, test_dataloaders=test_dataloader))

model = model.to('cpu')
