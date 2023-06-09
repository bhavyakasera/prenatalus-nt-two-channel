import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], testing=False):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck layer
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.testing = testing
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        skip_connections = []

        # Encoder: Down segments of UNET
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        if self.testing:
            x = self.dropout(x)
        x = self.bottleneck(x)
        if self.testing:
            x = self.dropout(x)
        skip_connections.reverse()

        # Decoder: Up segments of UNET
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])  # resize to match sizes

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        # Classifier
        x = self.final_conv(x)

        return x


activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


class UNETResNet18(nn.Module):
    def __init__(self, out_channels=2):
        super(UNETResNet18, self).__init__()
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder.conv1.register_forward_hook(get_activation('conv1'))
        self.encoder.layer1.register_forward_hook(get_activation('layer1'))
        self.encoder.layer2.register_forward_hook(get_activation('layer2'))
        self.encoder.layer3.register_forward_hook(get_activation('layer3'))
        self.encoder.layer4.register_forward_hook(get_activation('layer4'))

        features = [64, 128, 256, 512]

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        self.ups = nn.ModuleList()

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.ups.append(nn.ConvTranspose2d(features[0], features[0], kernel_size=2, stride=2))
        self.ups.append(DoubleConv(features[0], features[0]))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_layers = ['layer1', 'layer2', 'layer3', 'layer4']
        skip_connections = []

        resnet_output = self.encoder(x)
        x = activation['layer4']

        for layer in skip_layers:
            skip_connections.append(activation[layer])

        skip_connections.reverse()

        x = self.bottleneck(x)

        for idx in range(0, len(self.ups)-2, 2):
            x = self.ups[idx+1](x)
            skip_connection = skip_connections[idx//2]

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx](concat_skip)

        x = self.ups[-2](x)
        x = self.ups[-1](x)
        x = self.final_conv(x)

        return x


def test():
    x = torch.randn((2, 1, 160, 160))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)

    print(preds.shape)
    print(x.shape)

    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
