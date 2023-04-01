import torch
import os
import cv2
import numpy as np
from skimage.feature import canny
import shutil

def tensor2cv(imgs_t):
    imgs = []
    for i in range(imgs_t.size()[0]):
        im = imgs_t[i].cpu().detach().numpy().transpose((1, 2, 0))
        im = (im * 255).astype(np.uint8)
        imgs.append(im)

    return imgs

def cv2tensor(imgs):
    imgs_t = []
    for im in imgs:
        im_t = torch.from_numpy(im.transpose(2,0,1)).float().div(255.)
        im_t = torch.unsqueeze(im_t,dim=0)
        imgs_t.append(im_t)

    return torch.cat(imgs_t,dim=0)

def imgTensor2edge(imgs_t,sigma=2.,mask=None):
    imgs_np = tensor2cv(imgs_t)
    edge_imgs = []
    for im in imgs_np:
        gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        edge_im = canny(gray_im, sigma=sigma, mask=mask).astype(np.float)
        edge_im = np.expand_dims(edge_im,axis=-1)
        edge_imgs.append(edge_im)

    edge_imgs = cv2tensor(edge_imgs)
    return edge_imgs.to(imgs_t.device)

def checkDir(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def get_file_info(path):
    dir_path, file_full_name = os.path.split(path)
    file_name, file_type = os.path.splitext(file_full_name)

    return {"dir_path": dir_path, "file_name": file_name, "file_type": file_type}

