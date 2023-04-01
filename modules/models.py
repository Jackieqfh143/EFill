from modules.networks import *
import numpy as np

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,with_attn=False,nc_reduce=8):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//nc_reduce , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//nc_reduce , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,height,width  = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1) # B X (*W*H) X (C)
        proj_key =  self.key_conv(x).view(m_batchsize, -1, width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # B X (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,height,width)

        out = self.gamma*out + x

        if self.with_attn:
            return out,attention
        else:
            return out

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,):
        super().__init__()
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)

            cur_model = []
            cur_model += [
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]
            sequence.append(cur_model)

        nf_prev = nf
        nf = min(nf * 2, 512)

        cur_model = []
        cur_model += [
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]
        sequence.append(cur_model)

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, x):
        act = self.get_all_activations(x)
        return act[-1], act[:-1]

#augmented wit self-attention layer
class MultidilatedNLayerDiscriminatorWithAtt(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)

            cur_model = []
            cur_model += [
                MultidilatedConv(nf_prev, nf, kernel_size=kw, stride=2, padding=[2, 3, 6]),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]
            sequence.append(cur_model)

        nf_prev = nf
        nf = min(nf * 2, 512)

        cur_model = []
        cur_model += [
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]
        sequence.append(cur_model)

        cur_model = []
        cur_model += [
            Self_Attn(in_dim=nf, with_attn=False)
        ]

        sequence.append(cur_model)

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 3):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, x):
        act = self.get_all_activations(x)
        return act[-1], act[:-1]

class DistInpaintModel_SPADE_IN_LFFC_Base_concat_WithAtt(BaseNetwork):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9,merge_blocks=3,
                 norm_layer='bn',padding_mode='reflect', activation='relu',
                 out_act='tanh',max_features=512, nc_reduce = 2,ratio_g_in=0.5,ratio_g_out=0.5,
                 selected_edge_layers=[],selected_gt_layers=[],enable_lfu=False,is_training=False):
        assert (n_blocks >= 0)
        super().__init__()
        self.is_training = is_training
        self.num_down = n_downsampling
        self.merge_blks = merge_blocks
        self.n_blocks = n_blocks
        self.selected_edge_layers = selected_edge_layers
        self.selected_gt_layers = selected_gt_layers
        self.en_l0 = nn.Sequential( nn.ReflectionPad2d(2),
                NoFusionLFFC(in_channels=input_nc,out_channels=ngf,kernel_size=5,padding=0,
                             ratio_g_in=0,ratio_g_out=0,nc_reduce=1,norm_layer=norm_layer,activation=activation))

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == 0:
                g_in = 0
            else:
                g_in = ratio_g_in
            model = NoFusionLFFC(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 ratio_g_in=g_in,ratio_g_out=ratio_g_out,
                                 nc_reduce=1,norm_layer=norm_layer,activation=activation)
            setattr(self,f"en_l{i+1}",model)

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = TwoStreamLFFCResNetBlock(feats_num_bottleneck,nc_reduce=nc_reduce,ratio_g_in=ratio_g_in,
                                                    ratio_g_out=ratio_g_out,norm_layer=norm_layer,activation=activation)
            setattr(self, f"en_res_l{i}", cur_resblock)

        for i in range(merge_blocks):
            mult = 2 ** (n_downsampling - 1)
            model = LSPADEResnetBlock(fin=ngf * mult, fout=ngf * mult, semantic_nc=2 * ngf * mult,
                                          norm_layer='in')

            setattr(self, f"merge_l{i}", model)

        self.att = Self_Attn(in_dim=ngf * mult, with_attn=False)

        if self.is_training:
            ### structure decoder (only needed for training)
            for i in range(n_downsampling):
                mult = 2 ** (n_downsampling - i)
                model = GatedDeConv2d(min(max_features, ngf * mult),
                                             min(max_features, int(ngf * mult / 2)),
                                             kernel_size=3, stride=2, padding=1,
                                            norm_layer= norm_layer,activation=activation,
                                            mask_conv='normal', conv='normal'
                                        )

                setattr(self,f'edge_de_l{i}',model)

            edge_out_model = [nn.ReflectionPad2d(2),
                              GatedConv2dWithAct(ngf, 1, kernel_size=5, stride=1, padding=0,
                                                 norm_layer='', activation='', mask_conv='depthConv', conv='normal')]
            edge_out_model.append(actLayer(kind=out_act))
            self.edge_out_l = nn.Sequential(*edge_out_model)

        ### texture decoder
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i - 1)
            model = GatedDeConv2d(min(max_features, ngf * mult),
                                  min(max_features, int(ngf * mult / 2)),
                                  kernel_size=3, stride=2, padding=1,
                                  norm_layer=norm_layer, activation=activation,
                                  mask_conv='normal', conv='normal'
                                  )

            setattr(self, f'de_l{i}', model)

        out_model = [nn.ReflectionPad2d(2),
                  GatedConv2dWithAct(ngf // 2, output_nc, kernel_size=5, stride=1, padding=0,
                norm_layer='', activation='', mask_conv='depthConv', conv='normal')]
        out_model.append(actLayer(kind=out_act))
        self.out_l = nn.Sequential(*out_model)

        self.init_weights()

    def set_finetune_mode(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.InstanceNorm2d):
                module.eval()
            elif isinstance(module, nn.BatchNorm2d):
                module.eval()


    def forward(self, x):
        selected_edge_feats = {}
        selected_gt_feats = {}
        feats_dict = {}
        if self.is_training:
            for i in range(self.num_down + 1):
                x = getattr(self, f'en_l{i}')(x)
                x_l,x_g = x
                feats_dict[f'en_l{i}_xl'] = x_l
                feats_dict[f'en_l{i}_xg'] = x_g

            for i in range(self.n_blocks):
                x = getattr(self,f'en_res_l{i}')(x)
                x_l, x_g = x
                feats_dict[f'en_res_l{i}_xl'] = x_l
                feats_dict[f'en_res_l{i}_xg'] = x_g

            x_l, x_g = x
            x_l_, x_g_ = x_l.clone(),x_g.clone()
            edge = torch.cat((x_l_,x_g_),dim=1)
            edge2spade = edge
            feats_dict['edge2spade'] = edge
            for i in range(self.num_down):
                edge = getattr(self,f'edge_de_l{i}')(edge)
                feats_dict[f'edge_de_l{i}'] = edge

            merged_feat,edge_feat = x_l,edge2spade
            for i in range(self.merge_blks):
                merged_feat = getattr(self,f'merge_l{i}')(merged_feat,edge_feat)
                feats_dict[f'merge_l{i}'] = merged_feat

            att_x = self.att(merged_feat)
            x = att_x
            for i in range(self.num_down):
                x = getattr(self, f'de_l{i}')(x)
                feats_dict[f'de_l{i}'] = x

            out_edge = self.edge_out_l(edge)
            out_edge = (out_edge + 1) / 2.0
            selected_gt_feats = {k:feats_dict.get(k) for k in self.selected_gt_layers}
            selected_edge_feats = {k:feats_dict.get(k) for k in self.selected_edge_layers}
            feats_dict.clear()

        else:
            out_edge = None

            for i in range(self.num_down + 1):
                x = getattr(self, f'en_l{i}')(x)
                # feats_dict[f'en_l{i}'] = x

            for i in range(self.n_blocks):
                x = getattr(self, f'en_res_l{i}')(x)

            x_l, x_g = x
            x_l_, x_g_ = x_l.clone(), x_g.clone()
            edge = torch.cat((x_l_, x_g_), dim=1)
            edge2spade = edge
            # feats_dict['edge2spade'] = edge
            merged_feat, edge_feat = x_l, edge2spade
            for i in range(self.merge_blks):
                merged_feat = getattr(self, f'merge_l{i}')(merged_feat, edge_feat)

            att_x = self.att(merged_feat)
            x = att_x
            for i in range(self.num_down):
                x = getattr(self, f'de_l{i}')(x)

            # feats_dict.clear()

        out_x = self.out_l(x)
        out_x = (out_x + 1) / 2.0
        return out_x,out_edge,selected_edge_feats,selected_gt_feats


#use NoFusion LFFC as basic block
#inherently seperate local and global branch
#use the global output xg to learn holistic structure
#then inject it by SPADE into local detail generation
#nc_reduce = 2
#replace BN in SPAED with IN
#take concat(x_l,x_g) as the input of co-learning branch
#with self-attention layers
class Teacher_concat_WithAtt(BaseNetwork):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9,merge_blocks=3,
                 norm_layer='bn',padding_mode='reflect', activation='relu',
                 out_act='tanh',max_features=512, nc_reduce = 2,ratio_g_in=0.5,ratio_g_out=0.5,
                 selected_edge_layers=[],selected_gt_layers=[],enable_lfu=False,is_training=True):
        assert (n_blocks >= 0)
        super().__init__()
        self.is_training = is_training
        self.num_down = n_downsampling
        self.merge_blks = merge_blocks
        self.n_blocks = n_blocks
        self.selected_edge_layers = selected_edge_layers
        self.selected_gt_layers = selected_gt_layers
        self.en_l0 = nn.Sequential( nn.ReflectionPad2d(2),
                NoFusionLFFC(in_channels=input_nc,out_channels=ngf,kernel_size=5,padding=0,
                             ratio_g_in=0,ratio_g_out=0,nc_reduce=1,norm_layer=norm_layer,activation=activation))

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == 0:
                g_in = 0
            else:
                g_in = ratio_g_in
            model = NoFusionLFFC(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 ratio_g_in=g_in,ratio_g_out=ratio_g_out,
                                 nc_reduce=1,norm_layer=norm_layer,activation=activation)
            setattr(self,f"en_l{i+1}",model)

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = TwoStreamLFFCResNetBlock(feats_num_bottleneck,nc_reduce=nc_reduce,ratio_g_in=ratio_g_in,
                                                    ratio_g_out=ratio_g_out,norm_layer=norm_layer,activation=activation)
            setattr(self, f"en_res_l{i}", cur_resblock)

        for i in range(merge_blocks):
            mult = 2 ** (n_downsampling - 1)
            model = LSPADEResnetBlock(fin=ngf * mult, fout=ngf * mult, semantic_nc=2 * ngf * mult,
                                          norm_layer='in')

            setattr(self, f"merge_l{i}", model)

        self.att = Self_Attn(in_dim=ngf * mult, with_attn=False)

        if self.is_training:
            ### structure decoder (only needed for training)
            for i in range(n_downsampling):
                mult = 2 ** (n_downsampling - i)
                model = GatedDeConv2d(min(max_features, ngf * mult),
                                             min(max_features, int(ngf * mult / 2)),
                                             kernel_size=3, stride=2, padding=1,
                                            norm_layer= norm_layer,activation=activation,
                                            mask_conv='normal', conv='normal'
                                        )

                setattr(self,f'edge_de_l{i}',model)

            edge_out_model = [nn.ReflectionPad2d(2),
                              GatedConv2dWithAct(ngf, 1, kernel_size=5, stride=1, padding=0,
                                                 norm_layer='', activation='', mask_conv='normal', conv='normal')]
            edge_out_model.append(actLayer(kind=out_act))
            self.edge_out_l = nn.Sequential(*edge_out_model)

        ### texture decoder
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i - 1)
            model = GatedDeConv2d(min(max_features, ngf * mult),
                                  min(max_features, int(ngf * mult / 2)),
                                  kernel_size=3, stride=2, padding=1,
                                  norm_layer=norm_layer, activation=activation,
                                  mask_conv='normal', conv='normal'
                                  )

            setattr(self, f'de_l{i}', model)

        out_model = [nn.ReflectionPad2d(2),
                  GatedConv2dWithAct(ngf // 2, output_nc, kernel_size=5, stride=1, padding=0,
                norm_layer='', activation='', mask_conv='normal', conv='normal')]
        out_model.append(actLayer(kind=out_act))
        self.out_l = nn.Sequential(*out_model)

        self.init_weights()

    def set_finetune_mode(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.InstanceNorm2d):
                module.eval()
            elif isinstance(module, nn.BatchNorm2d):
                module.eval()


    def forward(self, x):
        selected_edge_feats = {}
        selected_gt_feats = {}
        feats_dict = {}
        if self.is_training:
            for i in range(self.num_down + 1):
                x = getattr(self, f'en_l{i}')(x)
                x_l,x_g = x
                feats_dict[f'en_l{i}_xl'] = x_l
                feats_dict[f'en_l{i}_xg'] = x_g

            for i in range(self.n_blocks):
                x = getattr(self,f'en_res_l{i}')(x)
                x_l, x_g = x
                feats_dict[f'en_res_l{i}_xl'] = x_l
                feats_dict[f'en_res_l{i}_xg'] = x_g

            x_l, x_g = x
            x_l_, x_g_ = x_l.clone(),x_g.clone()
            edge = torch.cat((x_l_,x_g_),dim=1)
            edge2spade = edge
            feats_dict['edge2spade'] = edge
            for i in range(self.num_down):
                edge = getattr(self,f'edge_de_l{i}')(edge)
                feats_dict[f'edge_de_l{i}'] = edge

            merged_feat,edge_feat = x_l,edge2spade
            for i in range(self.merge_blks):
                merged_feat = getattr(self,f'merge_l{i}')(merged_feat,edge_feat)
                feats_dict[f'merge_l{i}'] = merged_feat

            att_x = self.att(merged_feat)
            x = att_x
            for i in range(self.num_down):
                x = getattr(self, f'de_l{i}')(x)
                feats_dict[f'de_l{i}'] = x

            out_edge = self.edge_out_l(edge)
            out_edge = (out_edge + 1) / 2.0
            selected_gt_feats = {k:feats_dict.get(k) for k in self.selected_gt_layers}
            selected_edge_feats = {k:feats_dict.get(k) for k in self.selected_edge_layers}
            feats_dict.clear()

        else:
            out_edge = None

            for i in range(self.num_down + 1):
                x = getattr(self, f'en_l{i}')(x)
                feats_dict[f'en_l{i}'] = x

            for i in range(self.n_blocks):
                x = getattr(self, f'en_res_l{i}')(x)

            x_l, x_g = x
            x_l_, x_g_ = x_l.clone(), x_g.clone()
            edge = torch.cat((x_l_, x_g_), dim=1)
            edge2spade = edge
            feats_dict['edge2spade'] = edge
            merged_feat, edge_feat = x_l, edge2spade
            for i in range(self.merge_blks):
                merged_feat = getattr(self, f'merge_l{i}')(merged_feat, edge_feat)

            att_x = self.att(merged_feat)
            x = att_x
            for i in range(self.num_down):
                x = getattr(self, f'de_l{i}')(x)

            feats_dict.clear()

        out_x = self.out_l(x)
        out_x = (out_x + 1) / 2.0
        return out_x,out_edge,selected_edge_feats,selected_gt_feats


if __name__ == '__main__':
    from complexity import cuda_infer_time
    input1 = torch.randn(1, 4, 256, 256)
    input2 = torch.randn(1, 3, 256, 256)
    input3 = torch.randn(1,5,256,256)
    input4 = torch.randn(1,256,32,32)
    input5 = torch.randn(1, 256, 32, 32)
    import torchvision.transforms.functional as F
    dis_model = MultidilatedNLayerDiscriminatorWithAtt(input_nc=3)
    model9 = DistInpaintModel_SPADE_IN_LFFC_Base_concat_WithAtt(input_nc=5, output_nc=3, is_training=False, n_blocks=9)
    print_network_params(model9, "model9")

