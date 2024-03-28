import torch
import torchvision
import torch.nn.functional as F

from .clap import CustomCLAP
from .synthesizer_net import InnerProd
from .criterion import BCELoss, L1Loss, L2Loss, BinaryLoss
from .cond_unet_attn import CondAttUNet

def activate(x, activation):
    if activation == 'sigmoid':
        return torch.sigmoid(x)
    elif activation == 'softmax':
        return F.softmax(x, dim=1)
    elif activation == 'relu':
        return F.relu(x)
    elif activation == 'tanh':
        return F.tanh(x)
    elif activation == 'no':
        return x
    else:
        raise Exception('Unkown activation!')


class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.001)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.0001)

    def build_sound_net(self, fc_dim=64, weights='', args=None):
        net_sound = CondAttUNet(
            out_dim = fc_dim,
            cond_dim = args.cond_dim,
            num_downs = args.num_downs,
            num_cond_blocks = args.num_cond_blocks,
            num_res_layers = args.num_res_layers,
            num_head = args.num_head,
            cond_layer = args.cond_layer
        )        

        if len(weights) > 0:
            print('Loading weights for net_sound')
            net_sound.load_state_dict(torch.load(weights))

        return net_sound

    def build_custom_clap(self, enable_fusion=False, device=None, amodel= 'HTSAT-tiny', tmodel='roberta', channels = 32,
                    weights=''):

        net = CustomCLAP(enable_fusion=enable_fusion, device=device, amodel= amodel, tmodel=tmodel, channels = 32)

        if len(weights) > 0:
            print('Loading weights for net_condition')
            net.load_state_dict(torch.load(weights))

        return net

    def build_synthesizer(self, fc_dim=64, weights=''):
        net = InnerProd(fc_dim=fc_dim)

        net.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_synthesizer')
            net.load_state_dict(torch.load(weights))
        return net

    def build_criterion(self, arch):
        if arch == 'bce':
            net = BCELoss()
        elif arch == 'l1':
            net = L1Loss()
        elif arch == 'l2':
            net = L2Loss()
        elif arch == 'bn':
            net = BinaryLoss()
        else:
            raise Exception('Architecture undefined!')
        return net

