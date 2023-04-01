import torch.nn.functional as F
import random
import math
from complexity import  *

def actLayer(kind='relu'):
    if kind == 'tanh':
        return nn.Tanh()
    elif kind == 'sigmoid':
        return nn.Sigmoid()
    elif kind == 'relu':
        return nn.ReLU(inplace=True)
    elif kind == 'leaky':
        return nn.LeakyReLU(0.2,inplace=True)
    elif kind == 'elu':
        return nn.ELU(1.0, inplace=True)
    else:
        return nn.Identity()

def normLayer(channels,kind='bn',affine=True):
    if kind =='bn':
        return nn.BatchNorm2d(channels,affine=affine)
    elif kind == 'in':
        return nn.InstanceNorm2d(channels,affine=affine)
    else:
        return nn.Identity(channels)

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='kaiming', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                try:
                    nn.init.constant_(m.weight, 1)
                    nn.init.normal_(m.bias, 0.0001)
                except:
                    pass

        self.apply(init_func)


        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

    def print_networks(self,model_name):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()

        print('[Network %s] Total number of parameters : %.2f M' % (model_name, num_params / 1e6))
        print('-----------------------------------------------')

class SeperableConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1, groups=1,bias = True,padding_mode='reflect'):
        super(SeperableConv, self).__init__()
        self.depthConv = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups = in_channels,bias=bias, padding_mode=padding_mode)
        self.pointConv = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,groups=1,bias=bias, padding_mode=padding_mode)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self,x):
        x = self.depthConv(x)
        x = self.pointConv(x)

        return x

class MyConv2d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0,
           dilation=1, groups=1, bias=True,padding_mode='reflect',kind='depthConv',norm_layer='',activation = ''):
        super(MyConv2d, self).__init__()
        if kind == 'depthConv':
            self.conv = SeperableConv(in_channels, out_channels, kernel_size, stride, padding,
                                 dilation, groups, bias,padding_mode=padding_mode)
        else:
            self.conv =nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                 dilation, groups, bias,padding_mode=padding_mode)

        self.norm =normLayer(kind=norm_layer,channels=out_channels)
        self.act = actLayer(kind=activation)

    def forward(self,x):
        x = self.conv(x)
        x = self.act(self.norm(x))
        return x

class MyDeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,  stride=1, padding=0,
           dilation=1, groups=1, bias=True,padding_mode='reflect', kind='depthConv',scale_mode='bilinear',
                 norm_layer='',activation = ''):
        super(MyDeConv2d, self).__init__()
        if kind == 'depthConv':
            self.conv = SeperableConv(in_channels, out_channels, kernel_size, stride=1, padding=padding,
                                      dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding,
                                      dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

        self.norm = normLayer(kind=norm_layer, channels=out_channels)
        self.act = actLayer(kind=activation)

        self.scale_factor = stride
        self.scale_mode = scale_mode

    def forward(self, input):
        x = F.interpolate(input, scale_factor=self.scale_factor, mode=self.scale_mode)
        x = self.conv(x)
        x = self.act(self.norm(x))
        return x

#FU in paper
class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1,use_spectral=False,norm_layer='bn',activation='relu'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        #kernel size was fixed to 1
        #because the global receptive field.
        if not use_spectral:
            self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                              kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        else:
            self.conv_layer = nn.utils.spectral_norm(torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                              kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False))
        self.norm = normLayer(kind=norm_layer,channels=out_channels * 2)
        self.act = actLayer(kind=activation)

        nn.init.kaiming_normal_(self.conv_layer.weight)

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()

        #The FFT of a real signal is Hermitian-symmetric, X[i_1, ..., i_n] = conj(X[-i_1, ..., -i_n])
        # so the full fftn() output contains redundant information.
        # rfftn() instead omits the negative frequencies in the last dimension.

        # (batch, c, h, w/2+1) complex number
        ffted = torch.fft.rfftn(x, s=(h, w), dim=(2, 3), norm='ortho')  #norm='ortho' making the real FFT orthonormal
        ffted = torch.cat([ffted.real, ffted.imag], dim=1)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.act(self.norm(ffted))

        ffted = torch.tensor_split(ffted, 2, dim=1)
        ffted = torch.complex(ffted[0], ffted[1])
        output = torch.fft.irfftn(ffted, s=(h, w), dim=(2, 3), norm='ortho')

        return output

class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride = 1,groups=1, enable_lfu=True,norm_layer='bn',activation='relu'):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.enable_lfu = enable_lfu
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            normLayer(out_channels //2,kind=norm_layer),
            actLayer(kind=activation)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups,norm_layer=norm_layer,activation=activation)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups,norm_layer=norm_layer,activation=activation)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output

#light-weight version of original FFC
class NoFusionLFFC(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1,dilation=1, groups=1, bias=True,padding_mode='reflect',
                 norm_layer='bn', activation='relu',enable_lfu=False,ratio_g_in=0.5,ratio_g_out=0.5,nc_reduce=2,
                 out_act=True):
        super(NoFusionLFFC, self).__init__()
        self.ratio_g_in = ratio_g_in
        self.ratio_g_out = ratio_g_out
        in_cg = int(in_channels * ratio_g_in)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_g_out)
        out_cl = out_channels - out_cg

        if in_cl >0 and nc_reduce > 1:
            self.l_in_conv = nn.Sequential(
                nn.Conv2d(in_cl, in_cl // nc_reduce, kernel_size=1),
                normLayer(channels=in_cl // nc_reduce, kind=norm_layer),
                actLayer(kind=activation)
            )
        else:
            self.l_in_conv = nn.Identity()

        if out_cl >0 and nc_reduce >1:
            self.out_L_bn_act = nn.Sequential(
                nn.Conv2d(out_cl // nc_reduce, out_cl, kernel_size=1),
                normLayer(channels=out_cl, kind=norm_layer),
                actLayer(kind=activation if out_act else '')
            )
        elif out_cl >0:
            self.out_L_bn_act = nn.Sequential(
                normLayer(channels=out_cl, kind=norm_layer),
                actLayer(kind=activation if out_act else '')
            )
        else:
            self.out_L_bn_act = nn.Identity()

        if in_cg >0 and nc_reduce > 1:
            self.g_in_conv =  self.g_in_conv = nn.Sequential(
                nn.Conv2d(in_cg, in_cg // nc_reduce, kernel_size=1),
                normLayer(channels=in_cg // nc_reduce, kind=norm_layer),
                actLayer(kind=activation)
            )
        else:
            self.g_in_conv = nn.Identity()

        if out_cg >0 and nc_reduce > 1:
            self.out_G_bn_act = nn.Sequential(
                nn.Conv2d(out_cg // nc_reduce, out_cg, kernel_size=1),
                normLayer(channels=out_cg, kind=norm_layer),
                actLayer(kind=activation if out_act else '')
            )
        elif out_cg >0:
            self.out_G_bn_act = nn.Sequential(
                normLayer(channels=out_cg, kind=norm_layer),
                actLayer(kind=activation if out_act else '')
            )
        else:
            self.out_G_bn_act = nn.Identity()

        module = nn.Identity if in_cl == 0 or out_cl == 0 else SeperableConv
        self.convl2l = module(in_cl // nc_reduce, out_cl // nc_reduce, kernel_size,
                              stride, padding, dilation, groups, bias,padding_mode=padding_mode)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else SeperableConv
        self.convl2g = module(in_cl // nc_reduce, out_cg // nc_reduce, kernel_size,
                              stride, padding, dilation, groups, bias,padding_mode=padding_mode)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else SeperableConv
        self.convg2l = module(in_cg // nc_reduce, out_cl // nc_reduce, kernel_size,
                              stride, padding, dilation, groups, bias,padding_mode=padding_mode)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform

        self.convg2g = module(in_cg // nc_reduce, out_cg // nc_reduce, stride=stride,
            norm_layer=norm_layer,activation=activation, enable_lfu=enable_lfu)

        self.feats_dict = {}
        self.flops = 0

    def flops_count(self,module,input):
        if isinstance(module,nn.Module) and not isinstance(module,nn.Identity):
            if isinstance(input,torch.Tensor):
                # input_shape = input.shape[1:]
                flops = flop_counter(module,input)
                if flops != None:
                    self.flops += flops

    def get_flops(self):
        for m_name,input in self.feats_dict.items():
            module = getattr(self,m_name)
            self.flops_count(module,input)

        print(f'Total FLOPs : {self.flops:.5f} G')

        return self.flops

    def forward(self,x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0
        # self.feats_dict['l_in_conv'] = x_l
        # self.feats_dict['g_in_conv'] = x_g
        x_l,x_g = self.l_in_conv(x_l),self.g_in_conv(x_g)


        if self.ratio_g_out != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
            # self.feats_dict['convl2l'] = x_l
            # self.feats_dict['convg2l'] = x_g
            # self.feats_dict['out_L_bn_act'] = out_xl
            out_xl = self.out_L_bn_act(out_xl)
        if self.ratio_g_out != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)
            # self.feats_dict['convl2l'] = x_l
            # self.feats_dict['convg2l'] = x_g
            # self.feats_dict['out_G_bn_act'] = out_xg
            out_xg = self.out_G_bn_act(out_xg)

        return out_xl, out_xg

class GatedConv2dWithAct(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 norm_layer='bn', activation='relu', mask_conv='depthConv',conv ='normal',padding_mode='reflect'):
        super(GatedConv2dWithAct, self).__init__()
        self.conv2d = MyConv2d(in_channels, out_channels, kernel_size, stride, padding,
                                    dilation, groups, bias, padding_mode=padding_mode, kind=conv)
        self.mask_conv2d = MyConv2d(in_channels, out_channels, kernel_size, stride, padding,
                                    dilation, groups, bias, padding_mode=padding_mode, kind=mask_conv)
        self.norm = normLayer(kind=norm_layer, channels=out_channels)
        self.act = actLayer(kind=activation)
        self.gated = actLayer(kind='sigmoid')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        x = self.act(self.norm(x))
        x = x * self.gated(mask)

        return x

class GatedDeConv2d(torch.nn.Module):
    """
    Gated DeConvlution layer with activation (default activation:LeakyReLU)
    resize + conv
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, norm_layer='bn', activation='relu',mask_conv = 'depthConv',conv ='normal'):
        super(GatedDeConv2d, self).__init__()
        self.conv2d = GatedConv2dWithAct(in_channels, out_channels, kernel_size, stride=1, padding=padding,
                                         dilation=dilation, groups=groups, bias=bias, norm_layer=norm_layer,
                                         activation=activation,mask_conv=mask_conv,conv=conv)
        self.scale_factor = stride

    def forward(self, input):
        #print(input.size())
        x = F.interpolate(input, scale_factor=2)
        return self.conv2d(x)

class TwoStreamLFFCResNetBlock(nn.Module):
    def __init__(self,dim,kernel_size=3,padding=1,norm_layer='bn',activation='relu',nc_reduce=2,
                 ratio_g_in=0.5,ratio_g_out=0.5):
        super(TwoStreamLFFCResNetBlock, self).__init__()
        self.fusion = 0
        self.conv1 = NoFusionLFFC(in_channels=dim,out_channels=dim,kernel_size=kernel_size,padding=padding,
                                  norm_layer=norm_layer,activation=activation,out_act=True,nc_reduce=nc_reduce,
                                  ratio_g_in=ratio_g_in,ratio_g_out=ratio_g_out)
        self.conv2 = NoFusionLFFC(in_channels=dim, out_channels=dim, kernel_size=kernel_size,padding=padding,
                                  norm_layer=norm_layer, activation=activation, out_act=False, nc_reduce=nc_reduce,
                                  ratio_g_in=ratio_g_in,ratio_g_out=ratio_g_out)

    def get_flops(self):
        self.conv1.get_flops()
        self.conv2.get_flops()
        self.flops += self.conv1.flops + self.conv2.flops
        print(f'Total FLOPs : {self.flops:.5f} G')
        return self.flops


    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = self.conv1(x)
        x_l, x_g = self.conv2(x)

        out_x_l = x_l + id_l
        out_x_g = x_g + id_g
        return out_x_l, out_x_g

class LSPADE(nn.Module):
    def __init__(self, norm_nc, label_nc,norm_layer='bn'):
        super().__init__()

        self.param_free_norm = normLayer(norm_nc,kind=norm_layer,affine=False)
        # assert config_text.startswith('spade')
        # parsed = re.search('spade(\D+)(\d)x\d', config_text)
        # param_free_norm_type = str(parsed.group(1))
        # ks = int(parsed.group(2))
        #
        # if param_free_norm_type == 'instance':
        #     self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        # elif param_free_norm_type == 'syncbatch':
        #     self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        # elif param_free_norm_type == 'batch':
        #     self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        # else:
        #     raise ValueError('%s is not a recognized param-free norm type in SPADE'
        #                      % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        # pw = ks // 2
        self.mlp_shared = nn.Sequential(
            SeperableConv(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = SeperableConv(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = SeperableConv(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        # segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class LSPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, semantic_nc,norm_layer='bn'):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = SeperableConv(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = SeperableConv(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = SeperableConv(fin, fout, kernel_size=1, bias=False)

        # # apply spectral norm if specified
        # if 'spectral' in opt.norm_G:
        #     self.conv_0 = spectral_norm(self.conv_0)
        #     self.conv_1 = spectral_norm(self.conv_1)
        #     if self.learned_shortcut:
        #         self.conv_s = spectral_norm(self.conv_s)

        # # define normalization layers
        # spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = LSPADE(fin, semantic_nc,norm_layer)
        self.norm_1 = LSPADE(fmiddle, semantic_nc,norm_layer)
        if self.learned_shortcut:
            self.norm_s = LSPADE(fin, semantic_nc,norm_layer)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1,inplace=True)

class MultidilatedConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, dilation_num=3, comb_mode='sum', equal_dim=True,
                 shared_weights=False, padding=1, min_dilation=1, shuffle_in_channels=False, use_depthwise=False, **kwargs):
        super().__init__()
        convs = []
        self.equal_dim = equal_dim
        assert comb_mode in ('cat_out', 'sum', 'cat_in', 'cat_both'), comb_mode
        if comb_mode in ('cat_out', 'cat_both'):
            self.cat_out = True
            if equal_dim:
                assert out_dim % dilation_num == 0
                out_dims = [out_dim // dilation_num] * dilation_num
                self.index = sum([[i + j * (out_dims[0]) for j in range(dilation_num)] for i in range(out_dims[0])], [])
            else:
                out_dims = [out_dim // 2 ** (i + 1) for i in range(dilation_num - 1)]
                out_dims.append(out_dim - sum(out_dims))
                index = []
                starts = [0] + out_dims[:-1]
                lengths = [out_dims[i] // out_dims[-1] for i in range(dilation_num)]
                for i in range(out_dims[-1]):
                    for j in range(dilation_num):
                        index += list(range(starts[j], starts[j] + lengths[j]))
                        starts[j] += lengths[j]
                self.index = index
                assert(len(index) == out_dim)
            self.out_dims = out_dims
        else:
            self.cat_out = False
            self.out_dims = [out_dim] * dilation_num

        if comb_mode in ('cat_in', 'cat_both'):
            if equal_dim:
                assert in_dim % dilation_num == 0
                in_dims = [in_dim // dilation_num] * dilation_num
            else:
                in_dims = [in_dim // 2 ** (i + 1) for i in range(dilation_num - 1)]
                in_dims.append(in_dim - sum(in_dims))
            self.in_dims = in_dims
            self.cat_in = True
        else:
            self.cat_in = False
            self.in_dims = [in_dim] * dilation_num

        conv_type = SeperableConv if use_depthwise else nn.Conv2d
        dilation = min_dilation
        for i in range(dilation_num):
            if isinstance(padding, int):
                cur_padding = padding * dilation
            else:
                cur_padding = padding[i]
            convs.append(conv_type(
                self.in_dims[i], self.out_dims[i], kernel_size, padding=cur_padding, dilation=dilation, **kwargs
            ))
            if i > 0 and shared_weights:
                convs[-1].weight = convs[0].weight
                convs[-1].bias = convs[0].bias
            dilation *= 2
        self.convs = nn.ModuleList(convs)

        self.shuffle_in_channels = shuffle_in_channels
        if self.shuffle_in_channels:
            # shuffle list as shuffling of tensors is nondeterministic
            in_channels_permute = list(range(in_dim))
            random.shuffle(in_channels_permute)
            # save as buffer so it is saved and loaded with checkpoint
            self.register_buffer('in_channels_permute', torch.tensor(in_channels_permute))

    def forward(self, x):
        if self.shuffle_in_channels:
            x = x[:, self.in_channels_permute]

        outs = []
        if self.cat_in:
            if self.equal_dim:
                x = x.chunk(len(self.convs), dim=1)
            else:
                new_x = []
                start = 0
                for dim in self.in_dims:
                    new_x.append(x[:, start:start+dim])
                    start += dim
                x = new_x
        for i, conv in enumerate(self.convs):
            if self.cat_in:
                input = x[i]
            else:
                input = x
            outs.append(conv(input))
        if self.cat_out:
            out = torch.cat(outs, dim=1)[:, self.index]
        else:
            out = sum(outs)
        return out

class Multidilated_ResBlock(nn.Module):
    def __init__(self,channels,dilated_num=4,kernel_size=3,norm_layer='',activation='relu'):
        super(Multidilated_ResBlock, self).__init__()
        assert  channels % dilated_num ==0 and dilated_num > 0
        self.dilated_num = dilated_num
        for i in range(dilated_num):
            model = MyConv2d(in_channels=channels,out_channels=channels // dilated_num,kernel_size=kernel_size,
                             norm_layer=norm_layer,activation=activation,dilation=2**i,padding=2**i)
            setattr(self,f'block{i}',model)
        self.fuse_conv = MyConv2d(channels,channels,kernel_size=kernel_size,padding=1)

    def forward(self,x):
        residual = x
        out_x = []
        for i in range(self.dilated_num):
            out_x.append(getattr(self,f'block{i}')(x))
        out_x = torch.cat(out_x,dim=1)
        out_x = self.fuse_conv(out_x)
        out = out_x + residual
        return out


if __name__ == '__main__':
    x1 = torch.randn(1, 5, 256, 256)
    x2 = torch.randn(1, 256, 32, 32)
    model = Multidilated_ResBlock(channels=256)
    print_network_params(model,'model')
    flop_counter(model,x2)





