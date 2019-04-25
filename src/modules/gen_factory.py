from .layers import *


class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 which_conv=nn.Conv2d, which_bn=nn.InstanceNorm2d, activation=None,
                 upsample=None):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        self.upsample = upsample
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = self.which_bn(in_channels)
        self.bn2 = self.which_bn(out_channels)
        # upsample layers
        self.upsample = upsample

    def forward(self, x, y):
        h = self.activation(self.bn1(x))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(self.bn2(h))
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


class GBlockShallow(nn.Module):
    def __init__(self, in_channels, out_channels,
                 which_conv=nn.Conv2d, which_bn=None, activation=None,
                 upsample=None):
        super(GBlockShallow, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        self.upsample = upsample
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = self.which_bn(in_channels)
        # upsample layers
        self.upsample = upsample

    def forward(self, x, y):
        h = self.activation(self.bn1(x))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


class GBlockDeep(nn.Module):
    def __init__(self, in_channels, out_channels,
                 which_conv=nn.Conv2d, which_bn=None, activation=None,
                 upsample=None, channel_ratio=4):
        super(GBlockDeep, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.hidden_channels = self.in_channels // channel_ratio
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        self.upsample = upsample
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels,
                                     kernel_size=1, padding=0)
        self.conv2 = self.which_conv(self.hidden_channels, self.hidden_channels)
        self.conv3 = self.which_conv(self.hidden_channels, self.hidden_channels)
        self.conv4 = self.which_conv(self.hidden_channels, self.out_channels,
                                     kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = self.which_bn(self.in_channels)
        self.bn2 = self.which_bn(self.hidden_channels)
        self.bn3 = self.which_bn(self.hidden_channels)
        self.bn4 = self.which_bn(self.hidden_channels)
        # upsample layers
        self.upsample = upsample

    def forward(self, x, y):
        # Project down to channel ratio
        h = self.conv1(self.activation(self.bn1(x, y)))
        # Apply next BN-ReLU
        h = self.activation(self.bn2(h, y))
        # Drop channels in x if necessary
        if self.in_channels != self.out_channels:
            x = x[:, :self.out_channels]
            # Upsample both h and x at this point
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        # 3x3 convs
        h = self.conv2(h)
        h = self.conv3(self.activation(self.bn3(h, y)))
        # Final 1x1 conv
        h = self.conv4(self.activation(self.bn4(h, y)))
        return h + x


# Architectures for G
# Attention is passed in in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64.
def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    arch = {}
    arch[512] = {'in_channels':  [ch * item for item in [16, 16, 8, 8, 4, 2, 1]],
                 'out_channels': [ch * item for item in [16, 8, 8, 4, 2, 1, 1]],
                 'upsample':     [True] * 7,
                 'resolution':   [8, 16, 32, 64, 128, 256, 512],
                 'attention':    {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                                  for i in range(3, 10)}}
    arch[256] = {'in_channels':  [ch * item for item in [16, 16, 8, 8, 4, 2]],
                 'out_channels': [ch * item for item in [16, 8, 8, 4, 2, 1]],
                 'upsample':     [True] * 6,
                 'resolution':   [8, 16, 32, 64, 128, 256],
                 'attention':    {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                                  for i in range(3, 9)}}
    arch[128] = {'in_channels':  [ch * item for item in [16, 16, 8, 4, 2]],
                 'out_channels': [ch * item for item in [16, 8, 4, 2, 1]],
                 'upsample':     [True] * 5,
                 'resolution':   [8, 16, 32, 64, 128],
                 'attention':    {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                                  for i in range(3, 8)}}
    arch[64] = {'in_channels':  [ch * item for item in [16, 16, 8, 4]],
                'out_channels': [ch * item for item in [16, 8, 4, 2]],
                'upsample':     [True] * 4,
                'resolution':   [8, 16, 32, 64],
                'attention':    {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                                 for i in range(3, 7)}}
    arch[32] = {'in_channels':  [ch * item for item in [4, 4, 4]],
                'out_channels': [ch * item for item in [4, 4, 4]],
                'upsample':     [True] * 3,
                'resolution':   [8, 16, 32],
                'attention':    {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                                 for i in range(3, 6)}}

    return arch


class Generator(nn.Module):
    def __init__(self, ngf=64, nz=128, bottom_width=4, resolution=64,
                 G_kernel_size=3, G_attn='64',
                 activation='relu',
                 spectral=True, weight=False, norm_style='bn',
                 **kwargs):
        super(Generator, self).__init__()
        # Channel width mulitplier
        self.ngf = ngf
        # Dimensionality of the latent space
        self.nz = nz
        # The initial spatial dimensions
        self.bottom_width = bottom_width
        # Resolution of the output
        self.resolution = resolution
        # Kernel size?
        self.kernel_size = G_kernel_size
        # Attention?
        self.attention = G_attn
        # number of classes, for use in categorical conditional generation
        # nonlinearity for residual blocks
        self.activation = get_non_linearity(activation)

        self.spectral = spectral

        # Normalization style
        self.norm_style = norm_style

        # Architecture dict
        self.arch = G_arch(self.ngf, self.attention)[resolution]

        # Which convs, batchnorms, and linear layers to use

        self.which_conv = functools.partial(NormConv,
                                            kernel_size=3, padding=1, spectral=spectral, weight=weight)
        self.which_linear = functools.partial(NormLinear, spectral=spectral, weight=weight)

        self.which_bn = nn.BatchNorm2d

        # First linear layer
        self.linear = self.which_linear(nz,
                                        self.arch['in_channels'][0] * (self.bottom_width ** 2))

        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        # while the inner loop is over a given block
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[GBlock(in_channels=self.arch['in_channels'][index],
                                    out_channels=self.arch['out_channels'][index],
                                    which_conv=self.which_conv,
                                    which_bn=self.which_bn,
                                    activation=self.activation,
                                    upsample=(functools.partial(F.interpolate, scale_factor=2)
                                              if self.arch['upsample'][index] else None))]]

            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [Attention(self.arch['out_channels'][index], self.which_conv)]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        # output layer: batchnorm-relu-conv.
        # Consider using a non-spectral conv here
        final_conv = self.which_conv(self.arch['out_channels'][-1], 3)

        self.output_layer = nn.Sequential(nn.BatchNorm2d(num_features=self.arch['out_channels'][index]),
                                          self.activation,
                                          final_conv)

    # Note on this forward function: we pass in a y vector which has
    # already been passed through G.shared to enable easy class-wise
    # interpolation later. If we passed in the one-hot and then ran it through
    # G.shared in this forward function, it would be harder to handle.
    def forward(self, z, label=None):

        # First linear layer
        h = self.linear(z)
        # Reshape
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
        # print('init', h.shape)
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            for block in blocklist:
                h = block(h, label)
        # Apply batchnorm-relu-conv-tanh at output
        return torch.tanh(self.output_layer(h))



