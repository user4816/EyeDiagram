import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Residual Block with two convolutional layers."""
    def __init__(self, in_channels, out_channels, dilation=1):
        super(ResidualBlock, self).__init__()
        padding = dilation
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (CBAM)."""
    def __init__(self, in_channels, reduction=16, kernel_size=11):
        super(CBAMBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid_channel(out)
        x = x * out

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv_spatial(out)
        out = self.sigmoid_spatial(out)
        x = x * out
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512, 1024]):
        super(UNet, self).__init__()
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.encoder_layers.append(ResidualBlock(in_channels, feature))
            self.attention_layers.append(CBAMBlock(feature))
            in_channels = feature

        # Bottleneck with Extended Dilations
        self.bottleneck = nn.Sequential(
            ResidualBlock(features[-1], features[-1]*2, dilation=2),
            ResidualBlock(features[-1]*2, features[-1]*2, dilation=4),
            ResidualBlock(features[-1]*2, features[-1]*2, dilation=6)  # 추가된 dilation
        )

        # Decoder
        reversed_features = features[::-1]
        decoder_in_channels = features[-1]*2

        for feature in reversed_features:
            self.decoder_layers.append(
                nn.ConvTranspose2d(decoder_in_channels, feature, kernel_size=2, stride=2)
            )
            self.decoder_layers.append(nn.Sequential(
                ResidualBlock(feature * 2, feature),
                nn.Dropout(0.2)  
            ))
            decoder_in_channels = feature

        # Final Output Layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder with Attention
        for idx, layer in enumerate(self.encoder_layers):
            x = layer(x)
            x = self.attention_layers[idx](x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.decoder_layers), 2):
            x = self.decoder_layers[idx](x)  # Up-sampling
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])

            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder_layers[idx + 1](x)

        return self.final_conv(x)

if __name__ == "__main__":
    model = UNet()
    input_tensor = torch.randn(1, 3, 512, 512)
    output_tensor = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
