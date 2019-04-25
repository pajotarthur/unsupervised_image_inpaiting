import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils import spectral_norm as sn

from .non_local import SelfAttention


def snlinear(eps=1e-12, **kwargs):
    return sn(nn.Linear(**kwargs), eps=eps)


class Norm(nn.Module):
    def __init__(self, x_dim, z_dim, eps=1e-5, type_norm='batch'):
        """`x_dim` dimensionality of x input
           `z_dim` dimensionality of z latents
        """
        super(Norm, self).__init__()

        self.scale = snlinear(in_features=z_dim, out_features=x_dim, bias=False, eps=eps)
        self.offset = snlinear(in_features=z_dim, out_features=x_dim, bias=False, eps=eps)

    def forward(self, input, noise):
        # assumes input.dim() == 4, TODO: generalize that.
        if noise is not None:
            weight = 1 + self.scale(noise).unsqueeze(-1).unsqueeze(-1)
            bias = self.offset(noise).unsqueeze(-1).unsqueeze(-1)
            return input * weight + bias
        else:
            return input


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(
                nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(
                nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError(
                'normalization layer [%s] is not found' % norm_type)
    return norm_layer


# class BatchNormZ(nn.Module):
#     def __init__(self, in_channels, noise_size):
#         super().__init__()
#         self.batch_norm = nn.BatchNorm2d(in_channels, affine=False)
#         self.noise_emb = nn.Linear(noise_size, in_channels * 2)  # no sn here
#
#     def forward(self, input_features, noise):
#         result = self.batch_norm(input_features)
#         gamma, beta = self.noise_emb(noise).chunk(2, 1)  # 2 chunks along dim 1
#         gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # add H and W dimensions
#         beta = beta.unsqueeze(-1).unsqueeze(-1)
#         return gamma * result + beta


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, ngf=64, add_input=False, nz=0, n_blocks=9, n_down=1, padding_type='reflect', norm='batch',
                 self_attention=False, type_up='nearest', type_z='concat'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        norm_layer = get_norm_layer(norm_type=norm)
        self.n_down = n_down

        use_bias = False

        self.nz = nz
        self.num_classes = 0
        output_nc = input_nc
        self.self_attention = self_attention
        self.type_up = type_up
        self.type_z = type_z
        if type_z == 'concat':
            input_nc = input_nc + nz

        super(ResnetGenerator, self).__init__()
        self.add_input = add_input

        # self.replic1 = nn.ReflectionPad2d(3)
        # self.down1 = sn(nn.Conv2d(input_nc + nz, ngf, kernel_size=7, padding=0, bias=use_bias))
        # self.norm1 = norm_layer(ngf)

        model_down = [nn.ReflectionPad2d(3),
                      sn(nn.Conv2d(input_nc, ngf,
                                   kernel_size=7, padding=0, bias=use_bias)),
                      norm_layer(ngf),
                      nn.ReLU(True)]

        # mult = 1
        # self.down2 = sn(nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias))
        # self.norm2 = norm_layer(ngf * mult * 2)

        n_downsampling = n_down
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model_down += [sn(nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias)),
                           norm_layer(ngf * mult * 2),
                           nn.ReLU(True)]

        self.model_down = nn.Sequential(*model_down)
        model_block = []
        mult = 2 ** n_downsampling
        self.block1 = ResnetBlock(
                ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_bias=use_bias, nz=nz)
        self.block2 = ResnetBlock(
                ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_bias=use_bias, nz=nz)
        self.block3 = ResnetBlock(
                ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_bias=use_bias, nz=nz)
        self.block4 = ResnetBlock(
                ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_bias=use_bias, nz=nz)
        if self_attention:
            self.attention = SelfAttention(ngf * mult)
        self.block5 = ResnetBlock(
                ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_bias=use_bias, nz=nz)
        self.block6 = ResnetBlock(
                ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_bias=use_bias, nz=nz)

        self.up_1 = UpResnetBlock(
                ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_bias=use_bias, nz=nz, type_up=type_up)
        self.up_2 = UpResnetBlock(ngf * mult, padding_type=padding_type,
                                  norm_layer=norm_layer, use_bias=use_bias, nz=nz, type_up=type_up)
        model_up = []
        model_up += [nn.ReflectionPad2d(3)]
        model_up += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_up += [nn.Tanh()]

        self.model_up = nn.Sequential(*model_up)

    def forward(self, x, z=None, label=None, mask=None):
        """
        Standard forward
        """

        #
        # if self.type_z == 'concat':
        #     z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        #     h = torch.cat([x, z_img], 1)
        #
        # else:
        #     zz = (z.unsqueeze(1) * z.unsqueeze(-1))
        #     zz = zz.unsqueeze(1)
        #
        #     zz = zz.repeat(1, 1, x.size(2) // zz.size(2), x.size(3) // zz.size(3))
        #
        #     h = x + zz * mask[0, 0].type(torch.cuda.FloatTensor)

        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
            h = torch.cat([x, z_img], 1)
        else:
            h = x

        # z = None
        h = self.model_down(h)
        h = self.block1(h, z)

        h = self.block2(h, z)
        h = self.block3(h, z)
        h = self.block4(h, z)
        h = self.block5(h, z)
        h = self.block6(h, z)
        # h = self.up_1(h, z)
        h = self.up_2(h, z)
        h = self.model_up(h)

        return h

    def init_weights(self):
        # init fc to zero
        # self.fc.weight.data.zero_()
        # self.fc.bias.data.zero_()
        gain = 0

        blocks = [self.block1, self.block2, self.block3, self.block4, self.block5, self.block6, self.up_1, self.up_2]
        n_branches = len(blocks)

        for b in blocks:
            gain = 0
            for m in [b.norm2, b.conv2, b.norm1, b.conv1]:  # iterate in reverse order to init last conv with gain=0
                classname = m.__class__.__name__

                if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                    # print(classname, gain)
                    if gain == 0:
                        init.orthogonal_(m.weight.data, gain=0.002)
                    else:
                        init.orthogonal_(m.weight.data, gain=gain)

                    if hasattr(m, 'bias') and m.bias is not None:
                        init.constant_(m.bias.data, 0.0)

                    gain = n_branches ** (- 0.5)

                elif classname.find('Norm') != -1:

                    init.normal_(m.scale.weight.data, 1.0, gain)
                    if hasattr(m.scale, 'bias') and m.scale.bias is not None:
                        init.constant_(m.scale.bias.data, 0.0)
                    init.normal_(m.offset.weight.data, 1.0, gain)
                    if hasattr(m.offset, 'bias') and m.offset.bias is not None:
                        init.constant_(m.offset.bias.data, 0.0)

class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, upsample=None, type_up='transpose'):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample

        self.reflection_pad = nn.ReflectionPad2d(1)
        self.type_up = type_up

        if type_up == 'pixel':
            in_channels = in_channels // 4

        if type_up == 'transpose':
            self.conv = sn(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias))
        else:
            self.conv = sn(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=0, bias=bias))

        if type_up == 'pixel':
            self.interp = nn.PixelShuffle(upscale_factor=2)


    def forward(self, x):
        # print(x.shape)
        x_in = x
        if self.type_up == 'nearest':
            x_in = nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        elif self.type_up == 'bilinear':
            x_in = nn.functional.interpolate(x_in, mode='bilinear', scale_factor=self.upsample)
        elif self.type_up == 'pixel':
            x_in = self.interp(x_in)

        if self.type_up == 'transpose':
            out = x_in
        else:
            out = self.reflection_pad(x_in)

        out = self.conv(out)
        return out


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_bias, nz=6):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        p = 0
        if padding_type == 'reflect':
            self.replication_pad = nn.ReflectionPad2d(1)
        elif padding_type == 'replicate':
            self.replication_pad = nn.ReplicationPad2d(1)
        elif padding_type == 'zero':
            p = 1
            self.replication_pad = nn.Sequential([])
        else:
            raise NotImplementedError(
                    'padding [%s] is not implemented' % padding_type)

        self.conv1 = sn(nn.Conv2d(dim, dim, kernel_size=3,
                                  padding=p, bias=use_bias))
        self.norm1 = nn.BatchNorm2d(dim)
        # self.norm1 = Norm(dim, nz)
        self.conv2 = sn(nn.Conv2d(dim, dim, kernel_size=3,
                                  padding=p, bias=use_bias))
        # self.norm2 = Norm(dim, nz)
        self.norm2 = nn.BatchNorm2d(dim)

    def forward(self, x, z):
        """Forward function (with skip connections)"""
        h = self.replication_pad(x)
        h = self.conv1(h)
        h = self.norm1(h)
        h = F.relu(h)
        h = self.replication_pad(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.relu(h)
        return x + h


class UpResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_bias, nz=6, type_up='nearest', type_norm='batch'):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super().__init__()

        p = 0
        if padding_type == 'reflect':
            self.replication_pad = nn.ReflectionPad2d(1)
        elif padding_type == 'replicate':
            self.replication_pad = nn.ReplicationPad2d(1)
        elif padding_type == 'zero':
            p = 1
            self.replication_pad = nn.Sequential([])
        else:
            raise NotImplementedError(
                    'padding [%s] is not implemented' % padding_type)

        self.conv1 = UpsampleConvLayer(dim, int(
                dim / 2), kernel_size=3, stride=2, padding=0, bias=use_bias, upsample=2, type_up=type_up)
        self.conv_up = UpsampleConvLayer(dim, int(
                dim / 2), kernel_size=3, stride=2, padding=0, bias=use_bias, upsample=2, type_up=type_up)
        # self.norm1 = BatchNormZ(int(dim / 2), nz)
        # self.norm1 = Norm(int(dim / 2), nz)
        self.norm1 = nn.BatchNorm2d(int(dim / 2), nz)
        self.conv2 = sn(nn.Conv2d(int(dim / 2), int(dim / 2),
                                  kernel_size=3, padding=1, bias=use_bias))
        # self.norm2 = Norm(int(dim / 2), nz)
        self.norm2 = nn.BatchNorm2d(int(dim / 2), nz)

    def forward(self, x, z):

        h = self.conv1(x)
        h = self.norm1(h)
        h = F.relu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.relu(h)
        up_x = self.conv_up(x)
        return up_x + h