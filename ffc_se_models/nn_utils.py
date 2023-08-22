import torch

from improved_diffusion.unet import TimestepBlock


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


LRELU_SLOPE = 0.1


class AddSkipConn(TimestepBlock):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.add = torch.nn.quantized.FloatFunctional()

    def forward(self, x, t):
        return self.add.add(x, self.net(x, t))


class ConcatSkipConn(TimestepBlock):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, t):
        return torch.cat([x, self.net(x, t)], 1)
