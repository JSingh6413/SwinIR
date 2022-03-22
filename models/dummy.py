from torch import nn


class DummyNet(nn.Module):
    '''ONLY FOR DEBUG'''

    def __init__(self, in_channels=3):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=16,
                kernel_size=(3, 3),  padding=(1, 1)
            ), nn.ReLU(),
            nn.Conv2d(
                in_channels=16, out_channels=64,
                kernel_size=(3, 3), padding=(1, 1)
            ), nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=32,
                kernel_size=(3, 3), padding=(1, 1)
            ), nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=3,
                kernel_size=(3, 3), padding=(1, 1)
            ), nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)
