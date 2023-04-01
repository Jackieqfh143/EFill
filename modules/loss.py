from torch import nn
import torch
import os
import torchvision.models as models
import torch.nn.functional as F
from modules import resnet


def l1_loss(f1, f2, mask=1):
    return torch.mean(torch.abs(f1 - f2) * mask)

def smooth_l1_loss(f1,f2,mask=None):
    out = torch.nn.SmoothL1Loss(f1,f2)
    if mask != None:
        out = out * mask
    return out

def l2_loss(f1,f2,mask=None):
    out = F.mse_loss(f1,f2)
    if mask !=None:
        return out * mask
    return out

def featureMatchLoss(A_feats, B_feats,mask=None):
    assert len(A_feats) == len(B_feats)
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i].detach()
        B_feat = B_feats[i].detach()
        loss_value += torch.mean(torch.abs(A_feat - B_feat))
    return loss_value

def featureMatchLoss_mask(A_feats, B_feats,mask):
    assert len(A_feats) == len(B_feats)
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i].detach()
        B_feat = B_feats[i].detach()
        if A_feat.shape[3] != mask.shape[3]:
            mask_ = F.interpolate(mask, size=A_feat.shape[2:], mode='nearest')
        else:
            mask_ = mask

        loss_value += torch.mean(torch.abs(A_feat * mask_ - B_feat * mask_))      #only transfer knowledge in hole area
    return loss_value

def L2featureMatchLoss(target_feats, B_feats,mask=None):
    assert len(target_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(target_feats)):
        target_feat = target_feats[i].detach()
        B_feat = B_feats[i].detach()
        loss_value += F.mse_loss(B_feat,target_feat)
    return loss_value / len(target_feats)

def L2featureMatchLoss_mask(target_feats, B_feats,mask):
    #mask (0 for holes)
    assert len(target_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(target_feats)):
        target_feat = target_feats[i].detach()
        B_feat = B_feats[i].detach()
        if target_feat.shape[3] != mask.shape[3]:
            mask_ = F.interpolate(mask, size=target_feat.shape[2:], mode='nearest')
        else:
            mask_ = mask
        loss_value += F.mse_loss(B_feat * mask_ ,target_feat * mask_)     #only transfer knowledge in hole area
    return loss_value / len(target_feats)

def get_feat_mat_loss(type='l1'):
    if type == 'l1':
        return featureMatchLoss
    elif type == 'l1_mask':
        return featureMatchLoss_mask
    elif type == 'l2':
        return L2featureMatchLoss
    elif type == 'l2_mask':
        return L2featureMatchLoss_mask
    else:
        raise Exception('Unexpected loss type!')

#Non-saturate R1 GP (penalize only real data)
def make_r1_gp(discr_real_pred, real_batch):
    real_batch.requires_grad = True
    if torch.is_grad_enabled():
        grad_real = torch.autograd.grad(outputs=discr_real_pred.sum(), inputs=real_batch, create_graph=True)[0]
        grad_penalty = (grad_real.view(grad_real.shape[0], -1).norm(2, dim=1) ** 2).mean()
    else:
        grad_penalty = 0
    # real_batch.requires_grad = False

    return grad_penalty

def Dis_loss(pos, neg,type = 'Softplus',real_bt=None,lambda_r1=0.001):
    if type == 'Hinge':
        dis_loss = torch.mean(F.relu(1. - pos)) + torch.mean(F.relu(1. + neg))
    elif type == 'Softplus':
        dis_loss = F.softplus(- pos).mean() + F.softplus(neg).mean()
    elif type == 'R1':
        grad_penalty = make_r1_gp(pos, real_bt) *lambda_r1
        dis_loss = (F.softplus(- pos) + F.softplus(neg) + grad_penalty).mean()
    elif type == 'MSE':
        real_target = torch.zeros_like(pos)
        fake_target = torch.ones_like(neg)
        dis_loss = F.mse_loss(pos, real_target).mean() + F.mse_loss(neg, fake_target).mean()
    else:
        #BCE loss
        real_target = torch.zeros_like(pos)
        fake_target = torch.ones_like(neg)
        dis_loss = F.binary_cross_entropy(pos, real_target).mean() + F.binary_cross_entropy(neg, fake_target).mean()


    if type == 'R1':
        return dis_loss,grad_penalty.mean()
    else:
        return dis_loss,None

def Dis_loss_mask(pos,neg,mask,type = 'Softplus',real_bt=None,lambda_r1=0.001):
    if neg.shape[3] != mask.shape[3]:
        mask = F.interpolate(mask, size=neg.shape[2:], mode='nearest')

    #input mask (1 for fake part )
    if type == 'Hinge':
        dis_loss = F.relu(1. - pos) + mask * F.relu(1 + neg) + (1 - mask) * F.relu(1. - neg)
    elif type == 'Softplus':
        dis_loss = F.softplus(- pos) + F.softplus(neg) * mask + (1 - mask) * F.softplus(-neg)
    elif type == 'R1':
        grad_penalty = make_r1_gp(pos, real_bt) * lambda_r1
        dis_loss = F.softplus(- pos) + F.softplus(neg) * mask + (1 - mask) * F.softplus(-neg) + grad_penalty
    elif type == 'MSE':
        real_target = torch.zeros_like(pos)
        dis_loss = F.mse_loss(pos,real_target) + F.mse_loss(neg,mask)
    else:
        #BCE loss
        real_target = torch.zeros_like(pos)
        dis_loss = F.binary_cross_entropy(pos, real_target) + F.binary_cross_entropy(neg, mask)

    if type == 'R1':
        return dis_loss.mean(), grad_penalty.mean()
    else:
        return dis_loss.mean(),None

def Gen_loss(neg,type = 'Softplus'):
    if type == 'Hinge':
        gen_loss = -torch.mean(neg)
    elif type == 'Softplus' or type =='R1':
        gen_loss = F.softplus(-neg).mean()  #softplus is the smooth version of Relu()
    elif type == 'MSE':
        target = torch.zeros_like(neg)
        # MSE loss
        gen_loss = F.mse_loss(neg, target).mean()
    else:
        #BCE loss
        target = torch.zeros_like(neg)
        # BCE loss
        gen_loss = F.binary_cross_entropy(neg, target).mean()

    return gen_loss

def Gen_loss_mask(neg,mask,type = 'Softplus'):

    if neg.shape[3] != mask.shape[3]:
        mask = F.interpolate(mask,size=neg.shape[2:],mode='nearest')

    # input mask (1 for fake part )
    if type == 'Hinge':
        gen_loss = -neg * mask
    elif type == 'Softplus' or type =='R1':
        gen_loss = F.softplus(-neg)
        gen_loss = gen_loss * mask
    elif type == 'MSE':
        target = torch.zeros_like(neg)
        # MSE loss
        gen_loss = F.mse_loss(neg, target)
        gen_loss = gen_loss * mask
    else:
        target = torch.zeros_like(neg)
        #BCE loss
        gen_loss = F.binary_cross_entropy(neg,target)
        gen_loss = gen_loss * mask
    return gen_loss.mean()

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

# Model Builder
class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    @staticmethod
    def build_encoder(arch='resnet50dilated', fc_dim=512, weights=''):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()
        if arch == 'resnet18':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = resnet.Resnet(orig_resnet)
        elif arch == 'resnet18dilated':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = resnet.ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet50dilated':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = resnet.ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = resnet.Resnet(orig_resnet)
        else:
            raise Exception('Architecture undefined!')

        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def get_encoder(weights_path, arch_encoder, arch_decoder, fc_dim, segmentation,
                    *arts, **kwargs):
        if segmentation:
            path = os.path.join(weights_path, 'encoder_epoch_20.pth')
        else:
            path = ''
        return ModelBuilder.build_encoder(arch=arch_encoder, fc_dim=fc_dim, weights=path)

IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]

class ResNetPL(nn.Module):
    def __init__(self, weight=1,
                 weights_path=None, arch_encoder='resnet50dilated', segmentation=True):
        super().__init__()
        self.impl = ModelBuilder.get_encoder(weights_path=weights_path,
                                             arch_encoder=arch_encoder,
                                             arch_decoder='ppm_deepsup',
                                             fc_dim=2048,
                                             segmentation=segmentation)
        self.impl.eval()
        for w in self.impl.parameters():
            w.requires_grad_(False)

        self.weight = weight

    def forward(self, pred, target):
        pred = (pred - IMAGENET_MEAN.to(pred)) / IMAGENET_STD.to(pred)
        target = (target - IMAGENET_MEAN.to(target)) / IMAGENET_STD.to(target)

        pred_feats = self.impl(pred, return_feature_maps=True)
        target_feats = self.impl(target, return_feature_maps=True)

        result = torch.stack([F.mse_loss(cur_pred, cur_target)
                              for cur_pred, cur_target
                              in zip(pred_feats, target_feats)]).sum() * self.weight
        return result,target_feats,pred_feats