import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchvision.models import resnet50
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


class Deconv2dBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=False, dilation=1, batch_norm=True):
        super(Deconv2dBnRelu, self).__init__()
        self.batch_norm = batch_norm
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding,
                                         groups, bias, dilation)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.batch_norm:
            return self.relu(self.bn(self.deconv(x)))
        else:
            return self.relu(self.deconv(x))


class SimpleDecoder(nn.Module):
    def __init__(self, batch_norm=True, width=1, input_dim=20, output_act='sigmoid'):
        super(SimpleDecoder, self).__init__()
        self.conv = Deconv2dBnRelu(in_channels=input_dim, out_channels=256 * width, kernel_size=3, padding=0, stride=1)
        self.deconv1 = Deconv2dBnRelu(in_channels=256 * width, out_channels=128 * width, kernel_size=3, padding=1,
                                      stride=2,
                                      output_padding=1, batch_norm=batch_norm)
        self.deconv2 = Deconv2dBnRelu(in_channels=128 * width, out_channels=64 * width, kernel_size=3, padding=1,
                                      stride=2,
                                      output_padding=1, batch_norm=batch_norm)
        self.deconv3 = Deconv2dBnRelu(in_channels=64 * width, out_channels=32 * width, kernel_size=3, padding=1,
                                      stride=2,
                                      output_padding=1, batch_norm=batch_norm)
        self.deconv4 = Deconv2dBnRelu(in_channels=32 * width, out_channels=16 * width, kernel_size=3, padding=1,
                                      stride=2,
                                      output_padding=1, batch_norm=batch_norm)
        self.deconv5 = Deconv2dBnRelu(in_channels=16 * width, out_channels=8 * width, kernel_size=3, padding=1,
                                      stride=2,
                                      output_padding=1, batch_norm=batch_norm)
        self.conv6 = nn.Conv2d(in_channels=8 * width, out_channels=3, kernel_size=3, padding=1, bias=False)

        if output_act == 'id':
            self.out = nn.Identity()
        elif output_act == 'sigmoid':
            self.out = torch.sigmoid

    def forward(self, feature):
        feature = self.conv(feature[:, :, None, None])
        feature = self.deconv1(feature)
        feature = self.deconv2(feature)
        feature = self.deconv3(feature)
        feature = self.deconv4(feature)
        feature = self.deconv5(feature)
        pred_img = self.conv6(feature)
        pred_img = self.out(pred_img)
        return pred_img


class StampAE(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.decoder = SimpleDecoder(input_dim=20)
        self.encoder = resnet50(num_classes=20)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        x = batch  # x: binary code

        y = self.decoder(x)

        x_hat = torch.sigmoid(self.encoder(y))

        loss_bce = F.binary_cross_entropy(x_hat, x)
        loss_l2 = F.mse_loss(y, torch.zeros_like(y))

        loss = loss_bce + loss_l2 * 0.1
        log_data = {"step_bce_loss": loss_bce, "step_l2_loss": loss_l2}

        x_label = x.to(torch.bool)
        x_pred = (x_hat > 0.5)
        acc = x_pred.eq(x_label).to(torch.float).mean()
        log_data.update({"step_acc": acc})

        self.log_dict(log_data)

        return {"loss": loss}

    def configure_optimizers(self):
        parameters = list(self.decoder.parameters()) + list(self.encoder.parameters())
        optimizer = torch.optim.Adam(parameters, lr=3e-4, weight_decay=1e-3)
        return optimizer


class IndexDataset(Dataset):
    def __init__(self, size=105000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        code = np.array([int(x) for x in np.binary_repr(i + 1, width=20)], dtype=np.float32)
        return code


if __name__ == '__main__':
    from pytorch_lightning import Trainer

    train_dataloader = DataLoader(
        dataset=IndexDataset(),
        batch_size=256,
    )

    model = StampAE()
    trainer = Trainer(gpus=1)
    trainer.fit(model, train_dataloader)
