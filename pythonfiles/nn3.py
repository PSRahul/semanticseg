import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn


class pl_nn(pl.LightningModule):

    def __init__(self, hparams):

        super().__init__()

        self.save_hyperparameters(hparams)
        self.hparams = hparams

        filter_size_max = hparams['filter_size_max']

        self.drop = nn.Dropout(p=hparams['dropout_p'])

        self.relu = nn.PReLU()
        self.max = nn.MaxPool2d(2, ceil_mode=True)
        self.bn1_1 = nn.BatchNorm2d(filter_size_max//16)
        self.bn1_2 = nn.BatchNorm2d(filter_size_max//16)
        self.bn2_2 = nn.BatchNorm2d(filter_size_max//8)
        self.bn2_1 = nn.BatchNorm2d(filter_size_max//8)
        self.bn3_1 = nn.BatchNorm2d(filter_size_max//4)
        self.bn3_2 = nn.BatchNorm2d(filter_size_max//4)
        self.bn3_3 = nn.BatchNorm2d(filter_size_max//4)
        self.bn4_1 = nn.BatchNorm2d(filter_size_max//8)
        self.bn4_2 = nn.BatchNorm2d(filter_size_max//8)
        self.bn5_1 = nn.BatchNorm2d(filter_size_max//16)
        self.bn5_2 = nn.BatchNorm2d(filter_size_max//16)
        self.bn6 = nn.BatchNorm2d(filter_size_max//32)

        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1_1 = nn.Conv2d(3, filter_size_max//16, 3, padding=(1, 1))
        self.conv1_2 = nn.Conv2d(
            filter_size_max//16, filter_size_max//16, 3, padding=(1, 1))

        self.conv2_1 = nn.Conv2d(
            filter_size_max//16, filter_size_max//8, 3, padding=(1, 1))
        self.conv2_2 = nn.Conv2d(
            filter_size_max//8, filter_size_max//8, 3, padding=(1, 1))

        self.conv3_1 = nn.Conv2d(
            filter_size_max//8, filter_size_max//4, 3, padding=(1, 1))
        self.conv3_2 = nn.Conv2d(
            filter_size_max//4, filter_size_max//4, 3, padding=(1, 1))

        self.conv4_1 = nn.Conv2d(
            filter_size_max//4, filter_size_max//2, 3, padding=(1, 1))
        self.conv4_2 = nn.Conv2d(
            filter_size_max//2, filter_size_max//2, 3, padding=(1, 1))

        self.conv4_3 = nn.Conv2d(
            filter_size_max, filter_size_max//2, 3, padding=(1, 1))
        self.conv4_4 = nn.Conv2d(filter_size_max//2, filter_size_max//4, 3)

        self.conv3_3 = nn.Conv2d(
            filter_size_max//2, filter_size_max//4, 3, padding=(1, 1))
        self.conv3_4 = nn.Conv2d(
            filter_size_max//4, filter_size_max//8, 3, padding=(1, 1))

        self.conv2_3 = nn.Conv2d(
            filter_size_max//4, filter_size_max//8, 3, padding=(1, 1))
        self.conv2_4 = nn.Conv2d(
            filter_size_max//8, filter_size_max//16, 3, padding=(1, 1))

        self.conv1_3 = nn.Conv2d(
            filter_size_max//8, filter_size_max//16, 3, padding=(1, 1))
        self.conv1_4 = nn.Conv2d(
            filter_size_max//16, filter_size_max//32, 3, padding=(1, 1))

        self.conv1_5 = nn.Conv2d(filter_size_max//32, 3, 3, padding=(1, 1))

    def forward(self, x):

        x = self.bn1_1((self.relu(self.conv1_1(x))))
        x1 = self.bn1_2(self.relu(self.conv1_2(x)))
        x = self.drop(x)
        x = self.max(x1)

        x = self.bn2_1(self.relu(self.conv2_1(x)))
        x2 = self.bn2_2(self.relu(self.conv2_2(x)))
        x = self.drop(x)
        x = self.max(x2)

        x = self.bn3_1(self.relu(self.conv3_1(x)))
        x3 = self.bn3_2(self.relu(self.conv3_2(x)))
        x = self.drop(x)
        x = self.max(x3)

        x = self.up(x)
        x = torch.cat((x3, x), dim=1)
        x = self.bn3_3(self.relu(self.conv3_3(x)))
        x = self.bn4_1(self.relu(self.conv3_4(x)))

        x = self.drop(x)
        x = self.up(x)

        x = torch.cat((x2, x), dim=1)
        x = self.bn4_2(self.relu(self.conv2_3(x)))
        x = self.bn5_1(self.relu(self.conv2_4(x)))

        x = self.drop(x)
        x = self.up(x)

        x = torch.cat((x1, x), dim=1)
        x = self.bn5_2(self.relu(self.conv1_3(x)))
        x = self.bn6(self.relu(self.conv1_4(x)))

        x = self.conv1_5(x)

        return x

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.parameters(), weight_decay=self.hparams["weight_decay"])
        return optimizer

    def training_step(self, batch, batch_idx):

        inputs = batch["image"]
        labels = batch["label"]
        labels = 3*labels-1
        labels = labels.squeeze(1)
        labels = labels.to(dtype=torch.long)

        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)

        tensorboard_logs = {'train_loss_step': loss}

        return {'loss': loss,
                'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):

        inputs_val = batch["image"]
        labels_val = batch["label"]
        labels_val = 3*labels_val-1
        labels_val = labels_val.squeeze(1)
        labels_val = labels_val.to(dtype=torch.long)

        outputs_val = self(inputs_val)
        loss_val = F.cross_entropy(outputs_val, labels_val)

        return {'loss_val': loss_val}

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'train_loss': avg_loss, 'step': self.current_epoch}
        return {'loss': avg_loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss_val'] for x in outputs]).mean()
        tensorboard_logs = {'loss_val': avg_loss, 'step': self.current_epoch}
        return {'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):

        inputs_test = batch["image"]
        labels_test = batch["label"]
        labels_test = 3*labels_test-1
        labels_test = labels_test.squeeze(1)
        labels_test = labels_test.to(dtype=torch.long)

        outputs_test = self(inputs_test)
        loss_test = F.cross_entropy(outputs_test, labels_test)

        return {'loss_test': loss_test}

    def test_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss_test'] for x in outputs]).mean()
        tensorboard_logs = {'loss_test': avg_loss, 'step': self.current_epoch}
        return {'log': tensorboard_logs}
