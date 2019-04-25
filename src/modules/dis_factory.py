from .layers import *


# Residual block for the discriminator
class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv=nn.Conv2d, wide=True,
                 preactivation=False, activation=None, downsample=None, ):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv = which_conv
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample

        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)

    def shortcut(self, x):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def forward(self, x):
        if self.preactivation:
            # h = self.activation(x) # NOT TODAY SATAN
            # Andy's note: This line *must* be an out-of-place ReLU or it
            #              will negatively affect the shortcut connection.
            h = F.relu(x)
        else:
            h = x
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)

        return h + self.shortcut(x)


# Discriminator architecture, same paradigm as G's above
def D_arch(input_nc=3, ch=64, attention='64', ksize='333333', dilation='111111'):
    arch = {}
    arch[256] = {'in_channels':  [input_nc] + [ch * item for item in [1, 2, 4, 8, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
                 'downsample':   [True] * 6 + [False],
                 'resolution':   [128, 64, 32, 16, 8, 4, 4],
                 'attention':    {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                                  for i in range(2, 8)}}
    arch[128] = {'in_channels':  [input_nc] + [ch * item for item in [1, 2, 4, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 16]],
                 'downsample':   [True] * 5 + [False],
                 'resolution':   [64, 32, 16, 8, 4, 4],
                 'attention':    {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                                  for i in range(2, 8)}}
    arch[64] = {'in_channels':  [input_nc] + [ch * item for item in [1, 2, 4, 8]],
                'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
                'downsample':   [True] * 4 + [False],
                'resolution':   [32, 16, 8, 4, 4],
                'attention':    {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                                 for i in range(2, 7)}}
    arch[32] = {'in_channels':  [input_nc] + [item * ch for item in [4, 4, 4]],
                'out_channels': [item * ch for item in [4, 4, 4, 4]],
                'downsample':   [True, True, False, False],
                'resolution':   [16, 16, 16, 16],
                'attention':    {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                                 for i in range(2, 6)}}
    return arch


class Discriminator(nn.Module):

    def __init__(self, ndf=64, D_wide=True, resolution=64,
                 D_kernel_size=3, D_attn='32', activation='relu', output_dim=1,
                 spectral=True, weight=False, input_nc=3, **kwargs):
        super(Discriminator, self).__init__()
        # Width multiplier
        self.ch = ndf
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # Number of classes
        # Activation
        self.activation = get_non_linearity(activation)
        # Initialization style
        # Parameterization style

        # Architecture
        self.arch = D_arch(input_nc=input_nc,
                           ch=self.ch,
                           attention=self.attention)[resolution]

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        self.which_conv = functools.partial(NormConv,
                                            kernel_size=3, padding=1, spectral=spectral, weight=weight)
        self.which_linear = functools.partial(NormLinear, spectral=spectral, weight=weight)
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[DBlock(in_channels=self.arch['in_channels'][index],
                                    out_channels=self.arch['out_channels'][index],
                                    which_conv=self.which_conv,
                                    wide=self.D_wide,
                                    activation=self.activation,
                                    preactivation=(index > 0),
                                    downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
            # If attention on this block, attach it to the end
            # print(index, self.arch['attention'], self.arch['resolution'][index])
            if self.arch['attention'][self.arch['resolution'][index]]:
                # print('coucocucou')
                print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [Attention(self.arch['out_channels'][index],
                                              self.which_conv)]
        # exit()
        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)
        # Embedding for projection discrimination

    def forward(self, x, label=None):
        # Stick x into h for cleaner for loops without flow control
        h = x
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        # Apply global sum pooling as in SN-GAN
        h = torch.sum(self.activation(h), [2, 3])
        # Get initial class-unconditional output
        out = self.linear(h)
        # Get projection of final featureset onto class vectors and add to evidence
        return out
