import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        # todo
        self.Conv_down1 = downStep(1, 64)
        self.Conv_down2 = downStep(64, 128)
        self.Conv_down3 = downStep(128, 256)
        self.Conv_down4 = downStep(256, 512)
        self.Conv_down5 = downStep(512, 1024)
        self.Conv_up1 = upStep(1024, 512)
        self.Conv_up2 = upStep(512, 256)
        self.Conv_up3 = upStep(256, 128)
        self.Conv_up4 = upStep(128, 64)
        self.Conv_out = nn.Conv2d(64, n_classes, 1, padding=0, stride=1)
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!

    def forward(self, x):
        # todo
        x, conv1 = self.Conv_down1(x)
        x, conv2 = self.Conv_down2(x)
        x, conv3 = self.Conv_down3(x)
        x, conv4 = self.Conv_down4(x)
        _, x = self.Conv_down5(x)
        x = self.Conv_up1(x, conv4)
        x = self.Conv_up2(x, conv3)
        x = self.Conv_up3(x, conv2)
        x = self.Conv_up4(x, conv1)
        x = self.Conv_out(x)
        return x

class downStep(nn.Module):
    def __init__(self, inC, outC):

        super(downStep, self).__init__()
        # todo
        self.conv = nn.Sequential(nn.Conv2d(inC, outC, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(outC),
            nn.Conv2d(outC, outC, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(outC),)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # todo
        x = self.conv(x)
        x2 = self.pool(x)
        return x2,x

class upStep(nn.Module):
    def __init__(self, inC, outC, withReLU=True):
        super(upStep, self).__init__()
        # todo
        # Do not forget to concatenate with respective step in contracting path
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.transpose = nn.ConvTranspose2d(inC, outC, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(inC, outC, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(outC),
            nn.Conv2d(outC, outC, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(outC))

    def forward(self, x, x_down):
        # todo
        x = self.transpose(x)
        x_down = x_down[:, :, int((x_down.shape[2] - x.shape[2]) / 2):int((x_down.shape[2] + x.shape[2]) / 2),
                int((x_down.shape[3] - x.shape[2]) / 2):int((x_down.shape[3] + x.shape[2]) / 2)]
        x = torch.cat([x, x_down], dim=1)
        x = self.conv(x)
        return x