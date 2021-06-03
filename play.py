import torch.nn as nn
import torch


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
    def __init__(self, batch_norm=True, width=1, input_dim=20):
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

    def forward(self, feature):
        feature = self.conv(feature[:, :, None, None])
        feature = self.deconv1(feature)
        feature = self.deconv2(feature)
        feature = self.deconv3(feature)
        feature = self.deconv4(feature)
        feature = self.deconv5(feature)
        pred_img = torch.sigmoid(self.conv6(feature))
        return pred_img


if __name__ == '__main__':
    net = SimpleDecoder()
    x = torch.randn(10, 20)
    y = net(x)
    print(y.shape)
