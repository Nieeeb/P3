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
        self.blocks = nn.ModuleList([
            CANBlock(conv_channels, conv_channels, kernel_size=3, dilation=2 ** i)
            for i in range(num_blocks)
        ])
        self.output_conv = nn.Conv2d(conv_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.input_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_conv(x)
        return x
