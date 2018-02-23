import torch.nn as nn
from torch.nn import Sequential


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        n, m = x.size()[:2]
        return x.view(n, m)


def all_cnn_module():
    seq = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Conv2d(3, 96, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(96, 96, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(96, 96, 3, stride=2, padding=1),
        nn.ReLU(),

        nn.Dropout(p=0.5),
        nn.Conv2d(96, 192, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(192, 192, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(192, 192, 3, stride=2, padding=1),
        nn.ReLU(),

        nn.Dropout(p=0.5),
        nn.Conv2d(192, 192, 3, padding=0),
        nn.ReLU(),
        nn.Conv2d(192, 192, 1, padding=0),
        nn.ReLU(),
        nn.Conv2d(192, 10, 1, padding=0),
        nn.ReLU(),

        nn.AvgPool2d(6),
        Flatten()
    )
    return seq
