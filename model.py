import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
    """
    ResNet block: conv -> norm -> ReLU -> conv -> norm + skip connection
    """
    def __init__(self, dim, norm_layer=nn.InstanceNorm2d, use_dropout=False):

        super().__init__()
        block = []
        block += [nn.ReflectionPad2d(1),
                  nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                  norm_layer(dim),
                  nn.ReLU(True)]
        if use_dropout:
            block += [nn.Dropout(0.5)]
        block += [nn.ReflectionPad2d(1),
                  nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                  norm_layer(dim)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)

class ResnetGenerator(nn.Module):
  
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d,
                 use_dropout=False, n_blocks=9):
        assert n_blocks >= 0, "n_blocks must be non-negative"
        super().__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, bias=True),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2,
                                 kernel_size=3, stride=2, padding=1, bias=True),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer, use_dropout)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=True),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class NLayerDiscriminator(nn.Module):
    """
    Single-scale PatchGAN discriminator.
    """
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                   kernel_size=kw, stride=2,
                                   padding=padw, bias=True),
                         norm_layer(ndf * nf_mult),
                         nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                               kernel_size=kw, stride=1,
                               padding=padw, bias=True),
                     norm_layer(ndf * nf_mult),
                     nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv2d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1,
                               padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator that applies PatchGAN at multiple image scales.
    """
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d,
                 num_D=3):
        super().__init__()
        self.num_D = num_D
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1,
                                       count_include_pad=False)
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer)
            setattr(self, 'scale_{}'.format(i), netD)

    def forward(self, x):
        results = []
        input_downsampled = x
        for i in range(self.num_D):
            netD = getattr(self, 'scale_{}'.format(i))
            results.append(netD(input_downsampled))
            input_downsampled = self.downsample(input_downsampled)
        return results


def test_models():

    x = torch.randn(1, 3, 256, 256)
    
    G = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=9)
    y_hat = G(x)
    print('Generator output shape:', y_hat.shape)

    D = MultiScaleDiscriminator(input_nc=6, ndf=64, n_layers=3, num_D=3)
    d_outputs = D(torch.cat([x, y_hat], dim=1))
    for i, d_out in enumerate(d_outputs):
        print(f'Discriminator scale {i} output shape:', d_out.shape)


if __name__ == '__main__':
    test_models()
