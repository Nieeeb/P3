import torch
import torch.nn as nn

class CANBlock(nn.Module):
    """A single convolutional block in the Context Aggregation Network."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(CANBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=dilation, dilation=dilation, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.w0 = nn.Parameter(torch.tensor(1.0))  # Trainable weight
        self.w1 = nn.Parameter(torch.tensor(0.0))  # Trainable weight
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        convolved = self.conv(x)
        normalized = self.w0 * convolved + self.w1 * self.batch_norm(convolved)
        return self.activation(normalized)


class CANModel(nn.Module):
    """Context Aggregation Network Model."""
    def __init__(self, input_channels, conv_channels, out_channels, num_blocks):
        super(CANModel, self).__init__()
        self.input_conv = nn.Conv2d(input_channels, conv_channels, kernel_size=1, bias=False)
        dilation_rates = [1, 2, 4, 8, 16, 32, 64, 1]
        self.blocks = nn.ModuleList([
            CANBlock(conv_channels, conv_channels, kernel_size=3, dilation=rate)
            for rate in dilation_rates
        ])
        self.output_conv = nn.Conv2d(conv_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        o = self.input_conv(x)
        for block in self.blocks:
            o = block(o)
        o = self.output_conv(o)
        x = o + x
        return x
