# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Define the models."""
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


def init_weights(net):
    classname = net.__class__.__name__
    if classname.find("Conv") != -1:
        net.weight.data.normal_(0.0, 0.001)
    elif classname.find("BatchNorm") != -1:
        net.weight.data.normal_(1.0, 0.02)
        net.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        net.weight.data.normal_(0.0, 0.0001)


class SelfAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
    ):
        super().__init__()

        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            cond_dim,
            num_heads=8,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(cond_dim, dim * 2)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x, cond):
        _, condN, _ = cond.shape
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(cond).reshape(B, condN, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            act_layer=nn.GELU,
    ):
        super().__init__()

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x


class BlockSA(nn.Module):

    def __init__(
            self,
            dim,
            cond_dim,
            num_heads,
            mlp_ratio=2.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(
            dim,
            num_heads=num_heads,
        )

        self.norm2 = norm_layer(dim)

        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer
        )

    def forward(self, x, cond=None):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        x = x.permute(0, 2, 1).view(B, C, H, W)

        return x


class BlockIdentity(nn.Module):

    def __init__(
            self,
            dim,
            cond_dim,
            num_heads=None
    ):
        super().__init__()

    def forward(self, x, cond):
        return x

class BlockCA(nn.Module):

    def __init__(
            self,
            dim,
            cond_dim,
            num_heads,
            mlp_ratio=2.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim,
            cond_dim,
            num_heads=num_heads,
        )

        self.norm2 = norm_layer(dim)
        
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer
        )

    def forward(self, x, cond):
        if len(cond.shape) == 2:
            cond = cond.unsqueeze(1)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)
        x = x + self.attn(self.norm1(x), cond)
        x = x + self.mlp(self.norm2(x))

        x = x.permute(0, 2, 1).view(B, C, H, W)

        return x

class BlockSCA(nn.Module):

    def __init__(
            self,
            dim,
            cond_dim,
            num_heads,
            mlp_ratio=2.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
       
        self.sa_block = BlockSA(
            dim,
            cond_dim,
            num_heads,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
        )

        self.ca_block = BlockCA(
            dim,
            cond_dim,
            num_heads,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
        )

    def forward(self, x, cond):
        x = self.sa_block(x)
        x = self.ca_block(x, cond)

        return x


class BlockSCACat(nn.Module):

    def __init__(
            self,
            dim,
            cond_dim,
            num_heads,
            mlp_ratio=2.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
       
        self.sa_block = BlockSA(
            dim,
            num_heads,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
        )

        self.ca_block = BlockCA(
            dim,
            cond_dim,
            num_heads,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
        )

        self.cat_block = BlockCat(
            dim,
            cond_dim)

    def forward(self, x, cond):
        x = self.sa_block(x)
        x = self.cat_block(x, cond)
        x = self.ca_block(x, cond)
        return x


class BlockCat(nn.Module):

    def __init__(
            self,
            dim,
            cond_dim,
            num_heads=None
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                stride=1,
                padding=1
            )

        self.lin = nn.Linear(cond_dim, dim)
        
        self.conv2 = nn.Conv2d(
                dim*2,
                dim,
                kernel_size=3,
                stride=1,
                padding=1
            )

    def forward(self, x, cond):
        res = x

        x = self.conv1(x)
        
        B, C, H, W = x.size()

        cond = self.lin(cond)
        
        _, NC = cond.size()

        cond = cond.unsqueeze(-1).unsqueeze(-1) * torch.ones(
            (B, NC, H, W), device=x.device
        )

        x = torch.concat((x, cond), dim=1)

        x = self.conv2(x)

        x = x + res

        return x


class ResBlock(nn.Module):
    def __init__(
            self,
            input_nc,
            output_nc,
            num_res_layers=1
    ):
        super().__init__()

        assert input_nc == output_nc, "channel mismatched in resblock"

        self.num_res_layers = num_res_layers
        layers = []

        for i in range(num_res_layers):
            if i == num_res_layers-1:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(
                            output_nc,
                            output_nc,
                            kernel_size=3,
                            stride=1,
                            padding=1
                        ),
                        nn.BatchNorm2d(output_nc)
                    )
                )
            else:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(
                            output_nc,
                            output_nc,
                            kernel_size=3,
                            stride=1,
                            padding=1
                        ),
                        nn.BatchNorm2d(output_nc),
                        nn.LeakyReLU(0.2, True)
                    )
                )
        self.relu = nn.LeakyReLU(0.2, True)

        if num_res_layers == 0:
            self.res_layers = nn.Identity()
        else:        
            self.res_layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.num_res_layers == 0:
            return x
            
        res = x
        x = self.res_layers(x)        
        x = x + res
        x = self.relu(x)

        return x


class CondAttUNetBlockold(nn.Module):
    """A U-Net block that defines the submodule with skip connection.

    X ---------------------identity-------------------- X
      |-- downsampling --| submodule |-- upsampling --|

    """

    def __init__(
        self,
        outer_nc,
        inner_input_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        inner_output_nc=None,
        noskip=False,
        cond_nc=None,
        num_res_layers=3,
        num_cond_blocks=1,
        num_head=8,
        stride=2,
        cond_block=BlockCA
    ):
        super().__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.noskip = noskip
        self.cond_nc = cond_nc
        self.submodule = submodule

        use_bias = False
        if input_nc is None:
            input_nc = outer_nc
        if innermost:
            assert cond_nc > 0
            inner_output_nc = inner_input_nc
        elif inner_output_nc is None:
            inner_output_nc = 2 * inner_input_nc

        self.downnorm = nn.BatchNorm2d(inner_input_nc)
        self.uprelu = nn.ReLU(True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

        if outermost:
            self.downconv = nn.Conv2d(
                input_nc,
                inner_input_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            self.downresblock = ResBlock(
                inner_input_nc,
                inner_input_nc,
                num_res_layers=num_res_layers)

            self.upresblock = ResBlock(
                inner_output_nc,
                inner_output_nc,
                num_res_layers=num_res_layers
            )
            self.upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3, padding=1
            )

        elif innermost:
            self.downrelu = nn.LeakyReLU(0.2, True)
            self.downconv = nn.Conv2d(
                input_nc,
                inner_input_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )

            self.downresblock = ResBlock(
                inner_input_nc,
                inner_input_nc,
                num_res_layers=num_res_layers)

            self.ca_blocks = nn.Sequential(*[
                cond_block(
                    input_nc,
                    cond_nc,
                    num_head
                )
            for i in range(num_cond_blocks)]) 

            self.upresblock = ResBlock(
                inner_output_nc,
                inner_output_nc,
                num_res_layers=num_res_layers
            )

            self.upconv = nn.Conv2d(
                inner_output_nc,
                outer_nc,
                kernel_size=3,
                padding=1,
                bias=use_bias,
            )
            self.upnorm = nn.BatchNorm2d(outer_nc)

        else:
            self.downrelu = nn.LeakyReLU(0.2, True)
            self.downconv = nn.Conv2d(
                input_nc,
                inner_input_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            self.downresblock = ResBlock(
                inner_input_nc,
                inner_input_nc,
                num_res_layers=num_res_layers)

            self.upresblock = ResBlock(
                inner_output_nc,
                inner_output_nc,
                num_res_layers=num_res_layers
            )
            self.upconv = nn.Conv2d(
                inner_output_nc,
                outer_nc,
                kernel_size=3,
                padding=1,
                bias=use_bias,
            )
            self.upnorm = nn.BatchNorm2d(outer_nc)

    def forward(self, x, cond):
        if self.outermost:
            x_ = self.downconv(x)
            # x_ = self.downnorm(self.downresblock(x_))
            x_ = self.downresblock(x_)


            x_ = self.submodule(x_, cond)
            x_ = self.upresblock(self.upsample(self.uprelu(x_)))
            x_ = self.upconv(x_)

        elif self.innermost:
            x_ = self.downconv(self.downrelu(x))
            x_ = self.downresblock(x_)

            for block in self.ca_blocks:
                x_ = block(x_, cond)

            x_ = self.upresblock(self.upsample(self.uprelu(x_)))
            x_ = self.upnorm(self.upconv(x_))

        else:
            x_ = self.downnorm(self.downconv(self.downrelu(x)))
            x_ = self.downresblock(x_)
            x_ = self.submodule(x_, cond)
            x_ = self.upresblock(self.upsample(self.uprelu(x_)))
            x_ = self.upnorm(self.upconv(x_))
        
        if self.outermost or self.noskip:
            return x_
        else:
            return torch.cat([x, x_], 1)


class CondAttUNetBlock(nn.Module):
    """A U-Net block that defines the submodule with skip connection.

    X ---------------------identity-------------------- X
      |-- downsampling --| submodule |-- upsampling --|

    """

    def __init__(
        self,
        outer_nc,
        inner_input_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        inner_output_nc=None,
        noskip=False,
        cond_nc=None,
        num_res_layers=3,
        num_cond_blocks=1,
        num_head=8,
        stride=2,
        cond_block=BlockCA,
        bottom_condition=False
    ):
        super().__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.bottom_condition = bottom_condition
        self.noskip = noskip
        self.cond_nc = cond_nc
        self.submodule = submodule

        use_bias = False
        if input_nc is None:
            input_nc = outer_nc
        if innermost:
            assert cond_nc > 0
            inner_output_nc = inner_input_nc
        elif inner_output_nc is None:
            inner_output_nc = 2 * inner_input_nc

        self.downnorm = nn.BatchNorm2d(inner_input_nc)
        self.uprelu = nn.ReLU(True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

        if outermost:
            self.downconv = nn.Conv2d(
                input_nc,
                inner_input_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            self.downrelu = nn.LeakyReLU(0.2, True)

            self.downresblock = ResBlock(
                inner_input_nc,
                inner_input_nc,
                num_res_layers=num_res_layers)

            self.upresblock = ResBlock(
                inner_output_nc,
                inner_output_nc,
                num_res_layers=num_res_layers
            )
            self.upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3, padding=1
            )

        elif innermost:
            self.downrelu = nn.LeakyReLU(0.2, True)
            self.downconv = nn.Conv2d(
                input_nc,
                inner_input_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )

            self.downresblock = ResBlock(
                inner_input_nc,
                inner_input_nc,
                num_res_layers=num_res_layers)

            self.ca_blocks = nn.Sequential(*[
                cond_block(
                    input_nc,
                    cond_nc,
                    num_head
                )
            for i in range(num_cond_blocks)]) 

            self.upresblock = ResBlock(
                inner_output_nc,
                inner_output_nc,
                num_res_layers=num_res_layers
            )

            self.upconv = nn.Conv2d(
                inner_output_nc,
                outer_nc,
                kernel_size=3,
                padding=1,
                bias=use_bias,
            )
            self.upnorm = nn.BatchNorm2d(outer_nc)

        elif bottom_condition:
            self.downrelu = nn.LeakyReLU(0.2, True)
            self.downconv = nn.Conv2d(
                input_nc,
                inner_input_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )

            self.downresblock = ResBlock(
                inner_input_nc,
                inner_input_nc,
                num_res_layers=num_res_layers)

            self.ca_blocks = nn.Sequential(*[
                cond_block(
                    inner_input_nc,
                    cond_nc,
                    num_head
                )
            for i in range(num_cond_blocks)]) 

            self.upresblock = ResBlock(
                inner_output_nc,
                inner_output_nc,
                num_res_layers=num_res_layers
            )

            self.upconv = nn.Conv2d(
                inner_output_nc,
                outer_nc,
                kernel_size=3,
                padding=1,
                bias=use_bias,
            )
            self.upnorm = nn.BatchNorm2d(outer_nc)

        else:
            self.downrelu = nn.LeakyReLU(0.2, True)
            self.downconv = nn.Conv2d(
                input_nc,
                inner_input_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            self.downresblock = ResBlock(
                inner_input_nc,
                inner_input_nc,
                num_res_layers=num_res_layers)

            self.upresblock = ResBlock(
                inner_output_nc,
                inner_output_nc,
                num_res_layers=num_res_layers
            )
            self.upconv = nn.Conv2d(
                inner_output_nc,
                outer_nc,
                kernel_size=3,
                padding=1,
                bias=use_bias,
            )
            self.upnorm = nn.BatchNorm2d(outer_nc)

    def forward(self, x, cond):
        if self.outermost:
            x_ = self.downrelu(self.downnorm(self.downconv(x)))
            x_ = self.downresblock(x_)
            x_ = self.submodule(x_, cond)
            x_ = self.upresblock(self.upsample(x_))
            x_ = self.upconv(x_)

        elif self.innermost:
            x_ = self.downrelu(self.downnorm(self.downconv(x)))
            x_ = self.downresblock(x_)
            for block in self.ca_blocks:
                x_ = block(x_, cond)
            x_ = self.upresblock(self.upsample(x_))
            x_ = self.uprelu(self.upnorm(self.upconv(x_)))
        
        elif self.bottom_condition:
            x_ = self.downrelu(self.downnorm(self.downconv(x)))
            x_ = self.downresblock(x_)
            for block in self.ca_blocks:
                x_ = block(x_, cond)
            x_ = self.submodule(x_, cond)
            x_ = self.upresblock(self.upsample(x_))
            x_ = self.uprelu(self.upnorm(self.upconv(x_)))            
        else:
            x_ = self.downrelu(self.downnorm(self.downconv(x)))
            x_ = self.downresblock(x_)
            x_ = self.submodule(x_, cond)
            x_ = self.upresblock(self.upsample(x_))
            x_ = self.uprelu(self.upnorm(self.upconv(x_)))
        
        if self.outermost or self.noskip:
            return x_
        else:
            return torch.cat([x, x_], 1)



class CondAttUNet(nn.Module):
    """A UNet model."""

    def __init__(
        self,
        in_dim=1,
        out_dim=64,
        cond_dim=32,
        num_downs=5,
        ngf=64,
        use_dropout=False,
        num_res_layers=2,
        num_cond_blocks = 1,
        num_head = 8,
        cond_layer = 'ca'
    ):
        super().__init__()

        # Construct the U-Net structure
        if cond_layer == 'ca':
            Cond_Block = BlockCA
        elif cond_layer == 'sa':
             Cond_Block = BlockSA
        elif cond_layer == 'cat':
            Cond_Block = BlockCat
        elif cond_layer == 'sca':
            Cond_Block=BlockSCA
        elif cond_layer == 'sca_cat':
            Cond_Block=BlockSCACat
        elif cond_layer == 'identity':
            Cond_Block=BlockIdentity
        else:
            raise Exception("Conditioning block is not valid!")

        unet_block = CondAttUNetBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            innermost=True,
            cond_nc=cond_dim,
            num_cond_blocks=num_cond_blocks,
            num_res_layers=num_res_layers,
            num_head=num_head,
            cond_block=Cond_Block)

        for _ in range(num_downs - 5):
            unet_block = CondAttUNetBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                bottom_condition=True,
                cond_nc=cond_dim,
                num_cond_blocks=num_cond_blocks,
                num_res_layers=num_res_layers,
                num_head=num_head,
                cond_block=Cond_Block
            )

        unet_block = CondAttUNetBlock(
            ngf * 4, ngf * 8, input_nc=None, num_res_layers=num_res_layers, submodule=unet_block
        )
        unet_block = CondAttUNetBlock(
            ngf * 2, ngf * 4, input_nc=None, num_res_layers=num_res_layers, submodule=unet_block
        )
        unet_block = CondAttUNetBlock(
            ngf, ngf * 2, input_nc=None, num_res_layers=num_res_layers, submodule=unet_block
        )
        unet_block = CondAttUNetBlock(
            out_dim,
            ngf,
            input_nc=in_dim,
            num_res_layers=num_res_layers,
            submodule=unet_block,
            outermost=True,
        )

        self.bn0 = nn.BatchNorm2d(in_dim)
        self.unet_block = unet_block

    def forward(self, x, cond):
        x = self.bn0(x)
        x = self.unet_block(x, cond)
        return x