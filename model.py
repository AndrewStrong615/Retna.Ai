"""
Step 2: The neural network, written from scratch.

This is a ResNet-18-style convolutional neural network. Every weight starts
random: nothing is downloaded or pretrained. We use PyTorch's basic layers
(Conv2d, BatchNorm2d, Linear) the way every researcher does, but the
architecture, the wiring, and the initialization are all defined here.

Run `python model.py` to see how an image's shape changes through the network.
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    The core building block of a ResNet.

    The input x goes through two 3x3 convolutions, giving F(x). Then we ADD
    the original x back:

        output = ReLU( F(x) + x )

    That "+ x" is the skip (residual) connection. It gives the learning
    signal a direct path backward through the network, which is what lets
    deep networks train at all. It also means each block only has to learn
    a small correction to its input instead of a whole new representation.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # bias=False because BatchNorm right after has its own shift term.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # When this block shrinks the image (stride 2) or changes the number
        # of channels, x no longer has the same shape as F(x). A cheap 1x1
        # convolution reshapes x so the two can be added.
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class RetinaResNet(nn.Module):
    """
    stem   -> quickly shrinks the image 4x and detects simple features
    stage1 -> 64 feature maps at 1/4 resolution   (edges, small spots)
    stage2 -> 128 feature maps at 1/8 resolution  (vessel patterns, lesions)
    stage3 -> 256 feature maps at 1/16 resolution (clusters of lesions)
    stage4 -> 512 feature maps at 1/32 resolution (whole-eye patterns)
    pool   -> averages each feature map into one number
    head   -> combines those 512 numbers into one score (the "logit")
    """

    def __init__(self, blocks_per_stage=(2, 2, 2, 2), widths=(64, 128, 256, 512),
                 num_outputs=1, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        stages = []
        in_ch = widths[0]
        for i, (n_blocks, out_ch) in enumerate(zip(blocks_per_stage, widths)):
            stride = 1 if i == 0 else 2          # every stage after the first halves the size
            blocks = [BasicBlock(in_ch, out_ch, stride)]
            blocks += [BasicBlock(out_ch, out_ch) for _ in range(n_blocks - 1)]
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)       # randomly zeroes features during training to fight overfitting
        self.head = nn.Linear(in_ch, num_outputs)
        self._init_weights()

    def _init_weights(self):
        """
        Starting weights matter a lot when training from scratch. Too big and
        the signal explodes as it passes through layers; too small and it
        fades to nothing. Kaiming initialization scales the random weights
        so the signal keeps roughly the same strength through ReLU layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        # Start each block's residual branch at zero, so every block begins
        # as a simple pass-through. This makes early training more stable.
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.zeros_(m.bn2.weight)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.head(x).squeeze(1)           # one raw score per image


if __name__ == "__main__":
    model = RetinaResNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.1f} million (all random)\n")

    x = torch.randn(2, 3, 384, 384)              # a fake batch of 2 color images
    print(f"input            {tuple(x.shape)}")
    x = model.stem(x)
    print(f"after stem       {tuple(x.shape)}")
    for i, stage in enumerate(model.stages, 1):
        x = stage(x)
        print(f"after stage{i}     {tuple(x.shape)}")
    x = model.pool(x).flatten(1)
    print(f"after pooling    {tuple(x.shape)}")
    print(f"output           {tuple(model.head(x).squeeze(1).shape)}  <- one score per image")
