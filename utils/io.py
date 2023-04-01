import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
from PIL import Image



def get_state_dict_on_cpu(obj):
    cpu_device = torch.device('cpu')
    state_dict = obj.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(cpu_device)
    return state_dict


def save_ckpt(ckpt_name, models, optimizers, n_iter):
    ckpt_dict = {'n_iter': n_iter}
    for prefix, model in models:
        ckpt_dict[prefix] = get_state_dict_on_cpu(model)

    for prefix, optimizer in optimizers:
        ckpt_dict[prefix] = optimizer.state_dict()
    torch.save(ckpt_dict, ckpt_name)


def load_ckpt(ckpt_name, models, optimizers=None):
    ckpt_dict = torch.load(ckpt_name,map_location='cpu')
    for prefix, model in models:
        assert isinstance(model, nn.Module)
        model.load_state_dict(ckpt_dict[prefix], strict=False)
    if optimizers is not None:
        for prefix, optimizer in optimizers:
            optimizer.load_state_dict(ckpt_dict[prefix])
    return ckpt_dict['n_iter']


def load_network(model,save_dir,save_name):
    save_path = os.path.join(save_dir,save_name)
    model.load_state_dict(torch.load(save_path))
    return model

def save_network(state_dic,save_dir,save_name):
    save_path = os.path.join(save_dir,save_name)
    torch.save(state_dic,save_path)


class YamlHandler:

    def __init__(self,file,encoding = 'utf-8'):
         # assert os.path.exists(file)
         self.file = file
         self.encoding = encoding

    #读取yaml数据
    def get_ymal_data(self):
        with open(self.file,encoding=self.encoding) as f:
            data = yaml.load(f.read(),Loader=yaml.FullLoader)
        return data

    #写入yaml数据
    def write_yaml(self,data):
        with open(self.file,'w',encoding=self.encoding) as f:
            # set allow_unicode=True if contains any Chinese letters
            yaml.dump(data,stream=f,allow_unicode = True)


def save_img(img_tensor,save_dir,save_name,save_batch=False,toGray=False):
    if not save_batch:
        save_path = os.path.join(save_dir,save_name)
        img_tensor = img_tensor[0].cpu().detach().numpy().transpose((1,2,0))
        img_np = (img_tensor * 255).astype(np.uint8)
        if toGray:
            Image.fromarray(img_np).convert('L').save(save_path)
        else:
            Image.fromarray(img_np).save(save_path)
    else:
        for i in range(img_tensor.size(0)):
            save_path = os.path.join(save_dir, '({})_'.format(i) + save_name)
            img_t = img_tensor[i].cpu().detach().numpy().transpose((1,2,0))
            img_np = (img_t * 255).astype(np.uint8)
            if toGray:
                Image.fromarray(img_np).convert('L').save(save_path)
            else:
                Image.fromarray(img_np).save(save_path)


try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)
