import os
import shutil
import torch
import torch.nn as nn
import numpy as np
import importlib
from modules.baseModel import BaseModel
from modules.loss import ResNetPL,l1_loss,featureMatchLoss,Gen_loss,Gen_loss_mask,Dis_loss,\
    Dis_loss_mask,get_feat_mat_loss
from utils.util import checkDir
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from PIL import Image
from torch_ema import ExponentialMovingAverage

def find_model_using_name(model_name,**kwargs):
    model_filename = f"modules.models"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, nn.Module):
            model = cls
    if model !=None:
        return model(**kwargs)
    else:
        raise Exception(f'No model named {model_name} from model zoo!')

class InpaintingModel(BaseModel):
    def __init__(self,opt):
        super(InpaintingModel, self).__init__(opt)
        self.count = 0
        self.opt = opt
        self.mode = opt.mode
        self.lossNet = ResNetPL(weights_path=opt.lossNetDir) #segmentation network for calculate percptual loss
        self.flops = None
        self.device = self.accelerator.device
        self.lossNet = self.lossNet.to(self.device)
        self.recorder = SummaryWriter(self.log_path)
        self.current_lr = opt.lr
        self.current_d_lr = opt.d_lr

        if self.mode == 1:
            self.G_Net = find_model_using_name(opt.Generator,**opt.generator)
            if opt.enable_teacher:
                self.stTeacher = find_model_using_name(opt.ST_Teacher,**opt.stGNet)
                self.stTeacher = self.stTeacher.to(self.device)
                self.load_network(self.opt.st_TeacherPath, self.stTeacher)
            self.G_opt = torch.optim.AdamW(self.G_Net.parameters(), opt.lr,
                                      betas=(opt.beta_g_min, opt.beta_g_max))

            self.acc_args = [self.G_Net, self.G_opt]

            if self.opt.gan_loss:
                self.D_Net = find_model_using_name(opt.Discriminator,input_nc=opt.generator.output_nc)

                self.D_opt = torch.optim.AdamW(self.D_Net.parameters(), lr=opt.d_lr,
                                     betas=(opt.beta_d_min, opt.beta_d_max))
                self.acc_args.insert(1,self.D_Net)
                self.acc_args.append(self.D_opt)

            if opt.restore_training:
                self.load()

            if self.opt.enable_ema:
                self.ema_G = ExponentialMovingAverage(self.G_Net.parameters(), decay=0.995)
                self.ema_G.to(self.device)

        elif self.mode == 2:
            self.G_Net = find_model_using_name(opt.Generator,**opt.stGNet)
            self.edge_D_Net = find_model_using_name(opt.Discriminator,input_nc=1)
            self.gt_D_Net = find_model_using_name(opt.Discriminator, input_nc=3)
            self.G_opt = torch.optim.AdamW(self.G_Net.parameters(), opt.lr,
                                           betas=(opt.beta_g_min, opt.beta_g_max))
            self.edge_D_opt = torch.optim.AdamW(self.edge_D_Net.parameters(), lr=opt.d_lr,
                                           betas=(opt.beta_d_min, opt.beta_d_max))
            self.gt_D_opt = torch.optim.AdamW(self.gt_D_Net.parameters(), lr=opt.d_lr,
                                           betas=(opt.beta_d_min, opt.beta_d_max))

            if opt.restore_training:
                self.load()

            # args that should be prepared for accelerator
            self.acc_args = [self.G_Net, self.edge_D_Net,self.gt_D_Net, self.G_opt, self.edge_D_opt,self.gt_D_opt]

        self.lossDict = {}
        self.print_loss_dict = {}
        self.im_dict = {}
        self.val_im_dict = {}

    def train(self):
        if self.mode == 1:
            self.G_Net.train()
            if self.opt.enable_teacher:
                self.stTeacher.eval().requires_grad_(False).to(self.device)
            if self.opt.gan_loss:
                self.D_Net.train()

            self.G_Net.is_training = True
        elif self.mode == 2:
            self.G_Net.train()
            self.gt_D_Net.train()
            self.edge_D_Net.train()

    def eval(self):
        if self.mode == 1:
            self.G_Net.eval()
            self.G_Net.is_training = False
        elif self.mode == 2:
            self.G_Net.eval()
            self.G_Net.is_training = True  #the co-learning branch is required for teacher model

    def set_input(self,real_imgs,masks,edge_imgs):
        self.real_imgs = real_imgs
        self.edge_imgs = edge_imgs
        self.mask = masks[:, 2:3, :, :]
        input_im = real_imgs / 0.5 - 1
        input_im_masked = input_im * self.mask  # 0 for holes
        input_edge = edge_imgs * self.mask
        self.input = torch.cat((input_im_masked, self.mask,input_edge), dim=1)
        self.teacher_input = torch.cat((input_im,edge_imgs),dim=1)


    def forward(self,batch,count):
        self.count = count
        self.set_input(*batch)
        if self.mode == 1:
            if self.opt.enable_teacher:
                with torch.no_grad():
                    _, _, self.teach_edge_feats, self.teach_gt_feats = self.stTeacher(self.teacher_input)

            self.fake_imgs, self.fake_edges, self.fake_edge_feats, self.fake_gt_feats = self.G_Net(self.input)

            self.comp_imgs = self.real_imgs * self.mask + self.fake_imgs * (1 - self.mask)
            self.comp_edges = self.edge_imgs * self.mask + self.fake_edges * (1 - self.mask)


        elif self.mode == 2:
            self.fake_imgs, self.fake_edges, _, _ = self.G_Net( self.teacher_input)

    def backward_G(self):
        g_loss_list = []
        g_loss_name_list = []

        if self.opt.rec_loss:
            #reconstruct loss
            valid_l1_loss = self.opt.lambda_valid * (l1_loss(self.real_imgs, self.fake_imgs,self.mask) +
                                                     l1_loss(self.edge_imgs, self.fake_edges,self.mask))  # keep background unchanged
            hole_l1_loss = self.opt.lambda_hole * (l1_loss(self.real_imgs, self.fake_imgs, (1 - self.mask)) +
                                                   l1_loss(self.edge_imgs, self.fake_edges, (1 - self.mask)))
            rec_loss = valid_l1_loss + hole_l1_loss
            g_loss_list.append(rec_loss)
            g_loss_name_list.append("rec_loss")

        if self.opt.perc_loss:
            perc_loss1,_,_ = self.lossNet(self.comp_edges,self.edge_imgs)
            perc_loss2, _, _ = self.lossNet(self.comp_imgs, self.real_imgs)
            perc_loss = self.opt.lambda_perc * (perc_loss1 + perc_loss2)
            g_loss_list.append(perc_loss)
            g_loss_name_list.append("perc_loss")

        if self.opt.gan_loss:
            #adversarial loss & feature matching loss with gradient penalty
            dis_comp, comp_d_feats = self.D_Net(self.comp_imgs)

            gen_loss = Gen_loss_mask(dis_comp, (1 - self.mask), type=self.opt.gan_loss_type)

            gen_loss = self.opt.lambda_gen * gen_loss
            g_loss_list.append(gen_loss)
            g_loss_name_list.append("gen_loss")

            if self.opt.feat_mat:
                feat_mat_loss = self.opt.lambda_feat_mat * featureMatchLoss(self.real_d_feats, comp_d_feats)
                g_loss_list.append(feat_mat_loss)
                g_loss_name_list.append("feat_mat")

        # l1 distillation feature matching loss
        feat_mat_loss_func = get_feat_mat_loss(self.opt.feat_mat_loss_type)
        if self.opt.enable_teacher and self.opt.debug:
            self.accelerator.print('teach_edge_feats',self.teach_edge_feats.keys())
            self.accelerator.print('fake_edge_feats', self.fake_edge_feats.keys())
            self.accelerator.print('teach_gt_feats', self.teach_gt_feats.keys())
            self.accelerator.print('fake_gt_feats', self.fake_gt_feats.keys())

        if self.opt.enable_teacher:
            dist_edge_feat_loss = self.opt.lambda_dist_edge_feat * feat_mat_loss_func(list(self.teach_edge_feats.values()),
                                                                                list(self.fake_edge_feats.values()),1-self.mask)


            dist_gt_feat_loss = self.opt.lambda_dist_gt_feat * feat_mat_loss_func(list(self.teach_gt_feats.values()),
                                                                             list(self.fake_gt_feats.values()),1-self.mask)


            g_loss_list += [dist_edge_feat_loss, dist_gt_feat_loss]
            g_loss_name_list += ["dist_edge","dist_gt"]


        G_loss = 0.0
        for loss_name,loss in zip(g_loss_name_list,g_loss_list):
            self.lossDict[loss_name] = loss.item()
            G_loss += loss

        self.accelerator.backward(G_loss)

    def backward_D(self):
        if self.opt.gan_loss_type == 'R1':
            self.real_imgs.requires_grad = True

        dis_real, self.real_d_feats = self.D_Net(self.real_imgs)
        dis_comp, _ = self.D_Net(self.comp_imgs.detach())

        dis_loss,r1_loss = Dis_loss_mask(dis_real, dis_comp, (1 - self.mask), real_bt=self.real_imgs,
                                 type=self.opt.gan_loss_type,lambda_r1=self.opt.lambda_r1)

        self.lossDict['dis_loss'] = dis_loss.item()
        self.lossDict['r1_loss'] = r1_loss.item()
        self.accelerator.backward(dis_loss)

    def backward_dual_Teacher(self):
        g_loss_list = []
        g_loss_name_list = []

        if self.opt.rec_loss:
            # reconstruct loss
            rec_loss = self.opt.lambda_valid * (l1_loss(self.real_imgs, self.fake_imgs) +
                                                l1_loss(self.edge_imgs,self.fake_edges))
            g_loss_list.append(rec_loss)
            g_loss_name_list.append("rec_loss")

        if self.opt.perc_loss:
            # perceptual loss
            perc_loss1, _, _ = self.lossNet(self.fake_imgs, self.real_imgs)  # segmantation perception loss
            perc_loss2, _, _ = self.lossNet(self.fake_edges, self.edge_imgs)
            perc_loss = self.opt.lambda_perc * (perc_loss1 + perc_loss2)
            g_loss_list.append(perc_loss)
            g_loss_name_list.append("perc_loss")

        if self.opt.gan_loss:
            # adversarial loss & feature matching loss
            dis_fake, fake_d_feats = self.gt_D_Net(self.fake_imgs)
            dis_edge,fake_edge_d_feats = self.edge_D_Net(self.fake_edges)

            gen_loss = Gen_loss(dis_fake, type=self.opt.gan_loss_type) + Gen_loss(dis_edge,type=self.opt.gan_loss_type)
            gen_loss = self.opt.lambda_gen * gen_loss
            g_loss_list.append(gen_loss)
            g_loss_name_list.append("gen_loss")

            if self.opt.feat_mat:
                feat_mat_loss = self.opt.lambda_feat_mat * (featureMatchLoss(self.real_d_feats, fake_d_feats) +
                                                        featureMatchLoss(self.real_edge_d_feats,fake_edge_d_feats))
                g_loss_list.append(feat_mat_loss)
                g_loss_name_list.append("feat_mat_loss")

        G_loss = 0.0
        for loss_name, loss in zip(g_loss_name_list, g_loss_list):
            self.lossDict[loss_name] = loss.item()
            G_loss += loss

        self.accelerator.backward(G_loss)

    def backward_dual_DNet(self):
        if self.opt.gan_loss_type == 'R1':
            self.real_imgs.requires_grad = True
            self.edge_imgs.requires_grad = True
        dis_real, self.real_d_feats = self.gt_D_Net(self.real_imgs)
        dis_edge_real,self.real_edge_d_feats = self.edge_D_Net(self.edge_imgs)
        dis_fake, _ = self.gt_D_Net(self.fake_imgs.detach())
        dis_fake_edge,_ = self.edge_D_Net(self.fake_edges.detach())

        dis_loss1,r1_loss1  = Dis_loss(dis_real, dis_fake, real_bt=self.real_imgs, type=self.opt.gan_loss_type)
        dis_loss2, r1_loss2 = Dis_loss(dis_edge_real,dis_fake_edge,real_bt=self.edge_imgs,type=self.opt.gan_loss_type)
        dis_loss = dis_loss1 + dis_loss2

        self.lossDict['dis_loss'] = dis_loss.item()
        if r1_loss1 !=None:
            r1_loss = r1_loss1 + r1_loss2
            self.lossDict['r1_loss'] = r1_loss.item()
        self.accelerator.backward(dis_loss)

    def optimize_params(self):
        if self.mode == 1:
            if self.opt.gan_loss:
                with self.accelerator.accumulate(self.D_Net):
                    self.D_opt.zero_grad()
                    self.backward_D()
                    self.D_opt.step()

            with self.accelerator.accumulate(self.G_Net):
                self.G_opt.zero_grad()
                self.backward_G()
                if self.opt.use_grad_norm:
                    # gradient clip
                    self.accelerator.clip_grad_norm_(parameters=self.G_Net.parameters(),
                                                max_norm=self.opt.max_grad_norm,
                                                norm_type=self.opt.grad_norm_type)
                self.G_opt.step()
                if self.opt.enable_ema:
                    self.ema_G.update(self.G_Net.parameters())

        elif self.mode == 2:
            with self.accelerator.accumulate([self.edge_D_Net,self.gt_D_Net]):
                self.gt_D_opt.zero_grad()
                self.edge_D_opt.zero_grad()
                self.backward_dual_DNet()
                self.gt_D_opt.step()
                self.edge_D_opt.step()

            with self.accelerator.accumulate(self.G_Net):
                self.G_opt.zero_grad()
                self.backward_dual_Teacher()
                if self.opt.use_grad_norm:
                    # gradient clip
                    self.accelerator.clip_grad_norm_(parameters=self.G_Net.parameters(),
                                                     max_norm=self.opt.max_grad_norm,
                                                     norm_type=self.opt.grad_norm_type)
                self.G_opt.step()

        self.logging()

    # Adjust Learning rate
    def adjust_learning_rate(self, lr_in, min_lr, optimizer, epoch, lr_factor=0.95, warm_up=False, name='lr'):
        if not warm_up:
            lr = max(lr_in * lr_factor, float(min_lr))
        else:
            lr = max(lr_in * (epoch / int(self.opt.warm_up_epoch)), float(min_lr))

        print(f'Adjust learning rate to {lr:.5f}')
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        setattr(self, f'current_{name}', lr)

    @torch.no_grad()
    def validate(self,batch,count):
        self.val_count = count
        self.set_input(*batch)
        masked_imgs = None
        fake_edges = None
        if self.mode ==1:
            unwrap_model = self.accelerator.unwrap_model(self.G_Net)
            unwrap_model = unwrap_model.to(self.device)
            out, *_ = unwrap_model(self.input)
            fake_imgs = self.real_imgs * self.mask + out * (1 - self.mask)
            real_imgs = self.real_imgs
            masked_imgs = real_imgs * self.mask

        elif self.mode == 2:
            unwrap_model = self.accelerator.unwrap_model(self.G_Net)
            unwrap_model = unwrap_model.to(self.device)
            fake_imgs, fake_edges, _, _ = unwrap_model(self.teacher_input)
            real_imgs = self.real_imgs

        val_real_ims = []
        val_fake_ims = []
        val_masked_ims = []

        for i in range(fake_imgs.size()[0]):
            real_im = real_imgs[i].cpu().detach().numpy().transpose((1, 2, 0))
            real_im = (real_im * 255).astype(np.uint8)
            fake_im = fake_imgs[i].cpu().detach().numpy().transpose((1, 2, 0))
            fake_im = (fake_im * 255).astype(np.uint8)
            if masked_imgs !=None:
                masked_im = masked_imgs[i].cpu().detach().numpy().transpose((1, 2, 0))
                masked_im = (masked_im * 255).astype(np.uint8)
                val_masked_ims.append(masked_im)

            if fake_edges != None:
                fake_edge = fake_edges[i].cpu().detach().numpy().transpose((1, 2, 0))
                fake_edge = (fake_edge * 255).astype(np.uint8)
                val_masked_ims.append(fake_edge)

            val_real_ims.append(real_im)
            val_fake_ims.append(fake_im)

        if self.opt.record_val_imgs:
            self.val_im_dict['fake_imgs'] = fake_imgs.cpu().detach()
            self.val_im_dict['real_imgs'] = real_imgs.cpu().detach()
            if self.mode == 1:
                self.val_im_dict['masked_imgs'] = masked_imgs.cpu().detach()
            else:
                self.val_im_dict['fake_edges'] = fake_edges.cpu().detach()

        self.logging()

        return val_real_ims,val_fake_ims,val_masked_ims

    def get_current_imgs(self):
        if self.mode == 1:
            self.im_dict['real_imgs'] = self.real_imgs.cpu().detach()
            self.im_dict['masked_imgs'] = (self.real_imgs * self.mask).cpu().detach()
            self.im_dict['fake_imgs'] = self.fake_imgs.cpu().detach()
            self.im_dict['comp_imgs'] = self.comp_imgs.cpu().detach()

    def logging(self):
        for lossName, lossValue in self.lossDict.items():
            self.recorder.add_scalar(lossName, lossValue, self.count)

        if self.print_loss_dict == {}:
            temp = {k: [] for k in self.lossDict.keys()}
            self.print_loss_dict.update(temp)
            self.print_loss_dict['r1_loss'] = []
        else:
            for k, v in self.lossDict.items():
                if k in self.print_loss_dict.keys():
                    self.print_loss_dict[k].append(v)

        if self.opt.record_training_imgs:
            if self.count % self.opt.save_im_step == 0:
                self.get_current_imgs()
                for im_name, im in self.im_dict.items():
                    im_grid = vutils.make_grid(im, normalize=False, scale_each=True)
                    self.recorder.add_image(im_name, im_grid, self.count)

        if self.opt.record_val_imgs:
            if self.count % self.opt.val_step == 0:
                for im_name, im in self.val_im_dict.items():
                    im_grid = vutils.make_grid(im, normalize=False, scale_each=True)
                    self.recorder.add_image(im_name, im_grid, self.val_count)

    def reduce_loss(self):
        for k, v in self.print_loss_dict.items():
            if len(v) != 0:
                self.print_loss_dict[k] = sum(v) / len(v)
            else:
                self.print_loss_dict[k] = 0.0


    #save validate imgs
    def save_results(self,val_real_ims,val_fake_ims,val_masked_ims=None):
        im_index = 0
        val_save_dir = os.path.join(self.val_saveDir, 'val_results')
        if os.path.exists((val_save_dir)):
            shutil.rmtree(val_save_dir)
        checkDir([val_save_dir])
        if self.mode  == 1:
            for real_im, comp_im, masked_im in zip(val_real_ims, val_fake_ims, val_masked_ims):
                Image.fromarray(real_im).save(val_save_dir + '/{:0>5d}_im_truth.jpg'.format(im_index))
                Image.fromarray(comp_im).save(val_save_dir + '/{:0>5d}_im_out.jpg'.format(im_index))
                Image.fromarray(masked_im).save(val_save_dir + '/{:0>5d}_im_masked.jpg'.format(im_index))
                im_index += 1

        elif self.mode == 2:
            for real_im, fake_im,fake_edge in zip(val_real_ims, val_fake_ims,val_masked_ims):
                Image.fromarray(real_im).save(val_save_dir + '/{:0>5d}_im_truth.jpg'.format(im_index))
                Image.fromarray(fake_im).save(val_save_dir + '/{:0>5d}_im_out.jpg'.format(im_index))
                Image.fromarray(fake_edge[:, :, 0]).convert('L').save(val_save_dir + '/{:0>5d}_im_out_edge.jpg'.format(im_index))
                im_index += 1

    def load(self):
        if self.mode == 1:
            self.load_network(self.saveDir, self.G_Net, load_last=self.opt.load_last,load_from_iter=self.opt.load_from_iter)
            if self.opt.gan_loss:
                self.load_network(self.saveDir + '/latest_dis.pth', self.D_Net, load_last=self.opt.load_last)

        elif self.mode == 2:
            self.load_network(self.saveDir, self.G_Net, load_last=self.opt.load_last,load_from_iter=self.opt.load_from_iter)
            self.load_network(self.saveDir + '/latest_gt_dis.pth', self.gt_D_Net,
                              load_last=self.opt.load_last)
            self.load_network(self.saveDir + '/latest_edge_dis.pth', self.edge_D_Net,
                              load_last=self.opt.load_last)


    # save checkpoint
    def save_network(self,loss_mean_val,val_type='default'):

        src_save_path = os.path.join(self.saveDir, f"last_G_{val_type}.pth")

        save_path = os.path.join(self.saveDir,
                "G-step={}_lr={}_{}_loss={}.pth".format(self.count+1, round(self.current_lr,6), val_type,loss_mean_val))
        dis_save_path = os.path.join(self.saveDir, 'latest_dis.pth')

        self.accelerator.print('saving network...')


        #work for ditributed training
        # if self.opt.acc_save:
        if self.mode  == 1:
            os.rename(src_save_path,save_path)
            if self.opt.gan_loss:
                unwrap_model = self.accelerator.unwrap_model(self.D_Net)
                self.accelerator.save(unwrap_model.state_dict(), dis_save_path)

        elif self.mode == 2:
            os.rename(src_save_path, save_path)
            unwrap_model = self.accelerator.unwrap_model(self.gt_D_Net)
            self.accelerator.save(unwrap_model.state_dict(), dis_save_path.replace('_dis','_gt_dis'))

            unwrap_model = self.accelerator.unwrap_model(self.edge_D_Net)
            self.accelerator.save(unwrap_model.state_dict(), dis_save_path.replace('_dis', '_edge_dis'))

        self.accelerator.print('saving network done. ')





