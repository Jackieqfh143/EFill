from torch.utils.data import DataLoader
from evaluate.evaluation import validate
from prefetch_generator import BackgroundGenerator
from modules.models import DistInpaintModel_SPADE_IN_LFFC_Base_concat_WithAtt as EFill
from collections import OrderedDict
from tqdm import tqdm
import random
import warnings
import yaml
import argparse
import torch
import time
import os
import traceback
from torch.utils import data
from utils.im_process import *
from utils.visualize import show_im2_html
import torchvision.transforms.functional as F
import glob

parse = argparse.ArgumentParser()
parse.add_argument('--device',type=str,dest='device',default="cuda",help='device')
parse.add_argument('--dataset_name',type=str,dest='dataset_name',default="Place",help='dataset name')
parse.add_argument('--config_path',type=str,dest='config_path',default="./config/place_train.yaml",help='model config')
parse.add_argument('--model_path',type=str,dest='model_path',default="./checkpoints/place_best.pth",help='model path')
parse.add_argument('--mask_type',type=str,dest='mask_type',default="thick_256",help='the mask type')
parse.add_argument('--batch_size',type=int,dest='batch_size',default=8,help='batch size')
parse.add_argument('--target_size',type=int,dest='target_size',default=256,help='target image size')
parse.add_argument('--random_seed',type=int,dest='random_seed',default=2022,help='random seed')
parse.add_argument('--total_num',type=int,dest='total_num',default=10000,help='total number of test images')
parse.add_argument('--img_dir',type=str,dest='img_dir',default="",help='sample images for validation')
parse.add_argument('--mask_dir',type=str,dest='mask_dir',default="",help='sample masks for validation')
parse.add_argument('--save_dir',type=str,dest='save_dir',default="./masks",help='path for saving the masks')
parse.add_argument('--shuffle', action='store_true',help='shuffle the images')
arg = parse.parse_args()


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class ValDataSet(data.Dataset):
    def __init__(self,img_dir,mask_dir,total_num,shuffle=False):
        super(ValDataSet, self).__init__()
        self.imgs = sorted(glob.glob(img_dir + "/*.jpg") + glob.glob(img_dir + "/*.png"))
        self.masks = sorted(glob.glob(mask_dir + "/*.jpg") + glob.glob(mask_dir + "/*.png"))

        if shuffle:
            random.shuffle(self.imgs)
            random.shuffle(self.masks)

        max_num = min(len(self.imgs),total_num)
        self.imgs = self.imgs[:max_num]
        self.masks = self.masks[:max_num]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            traceback.print_exc()
            print('loading error: ' + self.imgs[index])
            item = self.load_item(0)

        return item

    def load_item(self,idx):
        input = self.preprocess(self.imgs[idx],self.masks[idx])
        return input

    #propocess one image each time
    def preprocess(self,img_path,mask_path):
        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)
            mask = np.concatenate((mask, mask, mask), axis=-1)

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
            img = np.concatenate((img, img, img), axis=-1)

        mask = mask.astype(np.uint8)
        sobel_edge = to_sobel(img)

        mask = (mask > 0).astype(np.uint8) * 255
        img_t_raw = F.to_tensor(img).float()
        edge_t = F.to_tensor(sobel_edge).float()
        mask_t = F.to_tensor(mask).float()

        mask_t = mask_t[0:1,:,:]
        mask_t = 1 - mask_t    #set holes = 0
        img_t = img_t_raw / 0.5 - 1
        masked_im = img_t * mask_t
        masked_edge = edge_t * mask_t
        input_x = torch.cat((masked_im,mask_t,masked_edge),dim=0)

        return input_x,img_t_raw,mask_t

def post_process(out,gt,mask,idx,save_path):
    masked_im = gt * mask
    comp_im = out * (1 - mask) + masked_im
    comp_img_np = []
    for i in range(comp_im.size(0)):
        gt_img_np = gt[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        mask_np = (1 - mask[i]).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        fake_img_np = comp_im[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        masked_img_np, _ = get_transparent_mask(gt_img_np, mask_np)  #the input mask should be 1 for holes
        Image.fromarray(masked_img_np).save(save_path + f'/{i + idx :0>5d}_im_masked.jpg')
        Image.fromarray(gt_img_np).save(save_path + f'/{i + idx:0>5d}_im_truth.jpg')
        Image.fromarray(fake_img_np).save(save_path + f'/{i + idx:0>5d}_im_out.jpg')

    return comp_img_np

def set_random_seed(random_seed=666,deterministic=False):
    if random_seed is not None:
        print("Set random seed as {}".format(random_seed))
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        if deterministic:
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            #for faster training
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

def load_model():
    with open(arg.config_path, encoding='utf-8') as f:
        opt = yaml.load(f.read(), Loader=yaml.FullLoader)
    model = EFill(**opt['generator'], is_training=False)
    net_state_dict = model.state_dict()
    state_dict = torch.load(arg.model_path, map_location='cpu')
    new_state_dict = {k: v for k, v in state_dict.items() if k in net_state_dict}
    model.load_state_dict(OrderedDict(new_state_dict), strict=False)
    model.eval().requires_grad_(False).to(arg.device)

    return model

if __name__ == '__main__':
    save_dir = os.path.join(arg.save_dir,arg.dataset_name)
    save_path = os.path.join(save_dir,arg.mask_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    set_random_seed(arg.random_seed)

    test_dataset = ValDataSet(arg.img_dir,arg.mask_dir,arg.total_num,arg.shuffle)

    test_dataloader = DataLoaderX(test_dataset,
                                 batch_size=arg.batch_size, shuffle=False, drop_last=False,
                                 num_workers=8,
                                 pin_memory=True)

    inpaintingModel = load_model()

    time_span = 0.0

    print("Processing images...")
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_dataloader)):
            input_x,gt,mask = batch
            input_x = input_x.to(arg.device)
            gt = gt.to(arg.device)
            mask = mask.to(arg.device)
            start_time = time.time()
            out,*_ = inpaintingModel(input_x)
            time_span += time.time() - start_time
            post_process(out,gt,mask,i * arg.batch_size,save_path)

    infer_speed = time_span / arg.total_num * 1000

    show_im2_html(web_title = f"Result_{arg.dataset_name}",
                  web_header = f"Inpainting Results on {arg.dataset_name}",
                  web_dir = save_dir,
                  img_dir = save_path,
                  im_size = arg.target_size,
                  max_num = 200)

    print("Start Validating...")
    validate(real_imgs_dir=save_path,
            comp_imgs_dir=save_path,
            device=arg.device,
            get_FID=True,
            get_LPIPS=True,
            get_IDS=True)

    print(f"Inference speed: {infer_speed} ms/ img")
