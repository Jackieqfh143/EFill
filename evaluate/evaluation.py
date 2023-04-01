import torch
import glob
import gc
import inspect
import time
import torchvision.transforms.functional as F
from tqdm import tqdm
from PIL import Image
import numpy as np
import lpips
import traceback
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader,Dataset
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from evaluate.fid import calculate_activation_statistics, calculate_frechet_distance
from evaluate.inception import InceptionV3
import sklearn.svm

def load_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    return img

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class miniDataset(Dataset):
    def __init__(self,real_img_dir,fake_img_dir,real_suffix=['_real'],fake_suffix=['_out']):
        super(miniDataset, self).__init__()
        self.real_imgs = self.load_list(real_img_dir,real_suffix)
        self.fake_imgs = self.load_list(fake_img_dir,fake_suffix)

    def __len__(self):
        return len(self.real_imgs)

    def __getitem__(self, index):
        try:
            items = self.load_item(index)
        except:
            traceback.print_exc()
            print('loading error: ' + self.real_imgs[index])
            return None
        return items

    def load_item(self,idx):
        real_im_path = self.real_imgs[idx]
        fake_im_path = self.fake_imgs[idx]
        real_im = Image.open(real_im_path).convert('RGB')
        fake_im = Image.open(fake_im_path).convert('RGB')

        return self.to_tensor(real_im),self.to_tensor(fake_im),real_im_path,fake_im_path

    def to_tensor(self, img):
        img_t = F.to_tensor(img).float()  # normalize to 0 ~ 1
        return img_t

    def load_list(self,img_path,suffix):
        if not isinstance(suffix,list):
            im_paths = sorted(glob.glob(img_path+f'/*{suffix}.jpg') + glob.glob(img_path+f'/*{suffix}.png'))
        else:
            im_paths = []
            for sf in suffix:
                im_paths += sorted(glob.glob(img_path+f'/*{sf}.jpg') + glob.glob(img_path+f'/*{sf}.png'))
        self.total_imgs = len(im_paths)
        return im_paths


def free_memory(to_delete: list):
    calling_namespace = inspect.currentframe().f_back
    for _var in to_delete:
        calling_namespace.f_locals.pop(_var, None)
        del_var(_var)
        torch.cuda.empty_cache()

def del_var(*args):
    for arg in args:
        del arg
        gc.collect()

def compare_mae(img_true, img_test):
    return np.mean(np.abs((np.mean(img_true, 2) - np.mean(img_test, 2)) / 255))

def ssim(frames1, frames2):
    error = 0
    for i in range(len(frames1)):
        im1 = load_img(frames1[i])
        im2 = load_img(frames2[i])
        error += compare_ssim(im1, im2, multichannel=True, win_size=11)
    return error


def psnr(frames1, frames2):
    error = 0
    for i in range(len(frames1)):
        im1 = load_img(frames1[i])
        im2 = load_img(frames2[i])
        error += compare_psnr(im1, im2)
    return error


def mae(frames1, frames2):
    error = 0
    for i in range(len(frames1)):
        im1 = load_img(frames1[i])
        im2 = load_img(frames2[i])
        error += compare_mae(im1, im2)
    return error

#take about at least 10k images to calculate the IDS
def cal_IDS(real_act,fake_act):
    svm = sklearn.svm.LinearSVC(dual=False)
    svm_inputs = np.concatenate([real_act, fake_act],axis=0)
    svm_targets = np.array([1] * real_act.shape[0] + [0] * fake_act.shape[0])
    print('Fitting ...')
    svm.fit(svm_inputs, svm_targets)                #this operation may take some time to run
    U_IDS = 1 - svm.score(svm_inputs, svm_targets)  #misclassification rate
    real_outputs = svm.decision_function(real_act)
    fake_outputs = svm.decision_function(fake_act)
    P_IDS = np.mean(fake_outputs > real_outputs)

    del_var(svm_inputs,svm_targets,svm,fake_act,fake_act)

    return float(U_IDS),float(P_IDS)

def get_act_stat(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu,sigma


def validate(real_imgs_dir,comp_imgs_dir,get_LPIPS=True,get_FID=True,
             get_IDS=False,batchSize=10,device='cuda:0',real_suffix=['_truth'],fake_suffix=['_out']):

    start_time = time.time()
    dataset = miniDataset(real_imgs_dir,comp_imgs_dir,real_suffix=real_suffix,fake_suffix=fake_suffix)
    data_loader = DataLoaderX(dataset,batch_size=batchSize,
                              shuffle=False,num_workers=8,drop_last=False)

    total_imgs = dataset.total_imgs
    # metrics_group = {'MAE':mae,'PSNR':psnr,'SSIM':ssim}
    # metrics_group = {'MAE': mae, 'SSIM': ssim}
    metrics_group = {}
    message_full = 'Current Performance: '
    scores = {}
    for key,val_method in metrics_group.items():
        print(f'\nCalculating {key}...')
        loss = 0.0
        for _,_,real_im_path,fake_im_path in tqdm(data_loader):
            loss += val_method(real_im_path,fake_im_path)

        loss = loss / total_imgs
        message_full += ' {}: {:.3f}'.format(key,loss)
        scores[key] = round(loss,4)


    # calculate lpips
    if get_LPIPS:
        print(f'\nCalculating LPIPS...')
        lpips_model = lpips.LPIPS(net='alex')
        lpips_model = lpips_model.to(device)
        lpips_ = 0.0
        for real_im,fake_im,_,_ in tqdm(data_loader):
            real_im = real_im.to(device)
            fake_im = fake_im.to(device)
            for i in range(real_im.shape[0]):
                loss = lpips_model.forward(real_im[i:i+1], fake_im[i:i+1],normalize=True)    #input img should be -1 ~ 1
                lpips_ += loss.detach().item()
        lpips_model.to('cpu')
        free_memory([lpips_model])
        lpips_ = lpips_ / total_imgs
        message_full += ' LPIPS: {:.3f}'.format(lpips_)
        scores['LPIPS'] = round(lpips_,4)

    #calculate fid
    if get_FID:
        print(f'\nCalculating FID...')
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]  # Choose the final average pooling features
        fid_model = InceptionV3([block_idx])
        fid_model.eval()
        fid_model = fid_model.to(device)

        with torch.no_grad():
            real_acts = []
            fake_acts = []
            for real_im,fake_im,_,_ in tqdm(data_loader):
                bt,c,h,w = real_im.shape
                real_im = real_im.to(device)
                fake_im = fake_im.to(device)
                real_pred = fid_model(real_im)[0]
                fake_pred = fid_model(fake_im)[0]
                real_pred = real_pred.cpu().reshape(bt,-1)
                fake_pred = fake_pred.cpu().reshape(bt,-1)
                if real_pred.shape[-1] == 2048:
                    real_acts.append(real_pred)

                if fake_pred.shape[-1] == 2048:
                    fake_acts.append(fake_pred)

            real_acts = torch.cat(real_acts,dim=0).numpy()
            fake_acts = torch.cat(fake_acts,dim=0).numpy()
            real_mu,real_sigma = get_act_stat(real_acts)
            fake_mu,fake_sigam = get_act_stat(fake_acts)
            fid_value = calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigam)
            message_full += ' FID: {:.3f}'.format(fid_value)
            scores['FID'] = round(fid_value,4)
            del_var(real_mu, real_sigma, fake_mu, fake_sigam)
            free_memory([fid_model])

            if get_IDS:
                print(f'\nCalculating IDS...')
                u_ids,p_ids = cal_IDS(real_acts,fake_acts)
                message_full += ' U-IDS: {:.3f}'.format(u_ids)
                scores['U-IDS'] = round(fid_value, 4)
                message_full += ' P-IDS: {:.3f}'.format(p_ids)
                scores['P-IDS'] = round(fid_value, 4)


    print('Finish evaluation!')

    loss_mean_ = round(loss_mean(scores), 4)
    time_span = time.time() - start_time
    message_full = message_full.replace('Current Performance: ','Current Performance: '+ f'loss_mean: {loss_mean_} ')
    print(message_full)
    print('Evaluate time span: ', time_span)

    return scores, loss_mean_,message_full

def loss_mean(metric_dict):
    metric_norms = {}
    loss = 0.0
    for k,v in metric_dict.items():
        metric_norms[k] = to_one_range(v)
    values = list(metric_norms.values())
    for k,v in metric_norms.items():
        if k in ['MAE','LPIPS','FID']:
            loss += v
        else:
            loss -= v
    loss = loss / len(values)
    return loss

import math

def to_one_range(x):
    x = ( 2 * math.atan(x)) / math.pi
    return x











