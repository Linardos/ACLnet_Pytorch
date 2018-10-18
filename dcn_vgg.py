from torch import nn
from torch.nn import MaxPool2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import Sigmoid, ReLU


def dcn_vgg(input_channels):

    model = nn.Sequential(

        Conv2d(input_channels, 64, kernel_size=(3, 3), padding=0),
        ReLU(),
        Conv2d(64, 64, kernel_size=(3, 3), padding=0),
        ReLU()
        MaxPool2d(kernel_size=(2,2), stride=(2,2))

        Conv2d(64, 128, kernel_size=(3, 3), padding=0),
        ReLU(),
        Conv2d(128, 128, kernel_size=(3, 3), padding=0),
        ReLU()
        MaxPool2d(kernel_size=(2,2), stride=(2,2))

        Conv2d(128, 256, kernel_size=(3, 3), padding=0),
        ReLU(),
        Conv2d(256, 256, kernel_size=(3, 3), padding=0),
        ReLU()
        MaxPool2d(kernel_size=(2,2), stride=(2,2))

        Conv2d(256, 512, kernel_size=(3, 3), padding=0),
        ReLU(),
        Conv2d(512, 512, kernel_size=(3, 3), padding=0),
        ReLU()
        MaxPool2d(kernel_size=(2,2), stride=(2,2))

        Conv2d(512, 512, kernel_size=(3, 3), padding=0),
        ReLU(),
        Conv2d(512, 512, kernel_size=(3, 3), padding=0),
        ReLU()
        Conv2d(512, 512, kernel_size=(3, 3), padding=0),
        ReLU()

        )

    return model
