import torch.nn as nn
from clstm import ConvLSTMCell
from dcn_vgg import dcn_vgg
from torch.nn import MaxPool2d
from torch.nn.modules.upsampling import Upsample
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import Sigmoid, ReLU

class TimeDistributed(nn.Module):

# An implementation made by a member of the community, it is an analogue of keras' TimeDistributed wrapper. https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/3

    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


def acl_vgg(data, stateful):
    dcn = dcn_vgg()
    att_module = nn.Sequential(
                    MaxPool2d(kernel_size=(2,2), stride=(2,2))
                    #The channels being input to maxpool are 512 (dcn output), to find the output channels of maxpool divide by the kernel size. 512/2=256
                    Conv2d(256, 64, kernel_size=(3, 3), padding=0)
                    ReLU()
                    Conv2D(64, 128, kernel_size=(3, 3), padding=0)
                    ReLU()
                    MaxPool2d(kernel_size=(2,2), stride=(2,2))
                    Conv2D(128, 64, kernel_size=(3, 3), padding=0)
                    ReLU()
                    Conv2D(64, 128, kernel_size=(3, 3), padding=0)
                    ReLU()
                    Conv2D(128, 1, kernel_size=(1, 1), padding=0)
                    Sigmoid()
                    Upsample(scale_factor=4, mode='nearest')
                    )


    outs = TimeDistributed(dcn)(data)

    attention = TimeDistributed(att_module)(outs)

    f_attention = TimeDistributed()(attention.view(attention.size()[0], -1)) #flatten
    f_attention = TimeDistributed()(f_attention.expand(512)) #repeatvector
    f_attention = TimeDistributed()(f_attention.transpose().unsqueeze(0)) #permute
    f_attention = TimeDistributed()(f_attention.reshape((32, 40, 512)))
    m_outs = outs * f_attention #elementwise multiplication
    outs = outs + m_outs

    ### This needs to change
    clstm = ConvLSTMCell(use_gpu=False, input_size=512, hidden_size=256, kernel_size=(3,3))
    outs = clstm(outs)
    ###

    produce_smaps = nn.Sequential(
                    #InputDimensions will be figured out after changing the ConvLSTM
                    Conv2D(InputDimensions, 1, kernel_size=(1, 1), padding=0)
                    Sigmoid()
                    Upsample(scale_factor=4, mode='nearest')
                    )

    outs = TimeDistributed(produce_smaps)(outs)
    attention = TimeDistributed(Upsample(scale_factor=2, mode='nearest'))(attention)
    return [outs, outs, outs, attention, attention, attention]
