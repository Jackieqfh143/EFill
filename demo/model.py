import torch
import yaml
import torch.nn as nn
from collections import OrderedDict
from im_process import *
from PIL import Image
import torchvision.transforms.functional as F
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.models import DistInpaintModel_SPADE_IN_LFFC_Base_concat_WithAtt as EFill

class InpaintingModel(nn.Module):
    def __init__(self,model_path,config_path):
        super(InpaintingModel, self).__init__()
        with open(config_path, encoding='utf-8') as f:
            opt = yaml.load(f.read(), Loader=yaml.FullLoader)

        self.model = EFill(**opt['generator'],is_training=False)
        net_state_dict = self.model.state_dict()
        state_dict = torch.load(model_path, map_location='cpu')
        new_state_dict = {k: v for k, v in state_dict.items() if k in net_state_dict}
        self.model.load_state_dict(OrderedDict(new_state_dict), strict=False)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.eval().requires_grad_(False).to(self.device)

    #propocess one image each time
    def preprocess(self,img,mask):
        sobel_edge = to_sobel(img)
        mask = (mask > 0).astype(np.uint8) * 255
        img_t = F.to_tensor(img).float()
        edge_t = F.to_tensor(sobel_edge).float()
        mask_t = F.to_tensor(mask).float()

        img_t_raw = torch.unsqueeze(img_t,dim=0)
        edge_t = torch.unsqueeze(edge_t, dim=0)
        mask_t = torch.unsqueeze(mask_t, dim=0)
        mask_t = mask_t[:,0:1,:,:]
        mask_t = 1 - mask_t    #set holes = 0
        img_t = img_t_raw / 0.5 - 1
        masked_im = img_t * mask_t
        masked_edge = edge_t * mask_t
        input_x = torch.cat((masked_im,mask_t,masked_edge),dim=1)
        self.GT = img_t_raw.to(self.device)
        self.mask = mask_t.to(self.device)

        return input_x

    def post_process(self,out_x):
        comp_im = out_x * (1 - self.mask) + self.GT * self.mask
        img_np = comp_im[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        return img_np

    def forward(self,x,mask):
        with torch.no_grad():
            masked_im = self.preprocess(x,mask)
            masked_im = masked_im.to(self.device)
            out_x,*_ = self.model(masked_im)

        return self.post_process(out_x)


if __name__ == '__main__':
    import time
    model_path = '../checkpoints/place_best.pth'
    config_path = './configs.yaml'
    test_img = './examples/imgs/1.jpg'
    test_mask = './examples/masks/1.png'
    model = InpaintingModel(model_path=model_path, config_path=config_path)
    img = np.array(Image.open(test_img))
    Image.fromarray(img).show()
    mask = np.array(Image.open(test_mask))
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=-1)
        mask = np.concatenate((mask, mask, mask), axis=-1)
    mask = mask.astype(np.uint8)
    Image.fromarray(mask).show()
    start_time = time.time()
    comp_im = model(img,mask)
    time_span = (time.time() - start_time) * 1000
    print('Inference time span : ',f'{time_span:.3f}ms')
    Image.fromarray(comp_im).show()