import argparse
from data.Lama_mask_gen import get_mask_generator
import os
import glob
from tqdm import tqdm
import random
from utils.im_process import *


parse = argparse.ArgumentParser()
parse.add_argument('--dataset_name',type=str,dest='dataset_name',default="Place",help='dataset name')
parse.add_argument('--mask_type',type=str,dest='mask_type',default="default",help='medium_256/medium_512/thick_256/thick_512/thin_256/thin_512/segm_256/segm_512')
parse.add_argument('--target_size',type=int,dest='target_size',default=256,help='target image size')
parse.add_argument('--aspect_ratio_kept', action='store_true',help='keep the image aspect ratio when resize it')
parse.add_argument('--fixed_size', action='store_true',help='fixed the crop size')
parse.add_argument('--center_crop', action='store_true',help='center crop')
parse.add_argument('--total_num',type=int,dest='total_num',default=10000,help='total number of the masks')
parse.add_argument('--img_dir',type=str,dest='img_dir',default="",help='sample images for validation')
parse.add_argument('--save_dir',type=str,dest='save_dir',default="./masks",help='path for saving the masks')
parse.add_argument('--shuffle', action='store_true',help='shuffle the images')
parse.add_argument('--max_obj_area',type=float,dest='max_obj_area',default=0.4,help='work for segm mask, the maximum covered area of the image')
parse.add_argument('--max_obj_num',type=int,dest='max_obj_num',default=2,help='work for segm mask, the maximum detected objects in the image')

arg = parse.parse_args()


def resize(img):
    if arg.aspect_ratio_kept:
        imgh, imgw = img.shape[0:2]
        side = np.minimum(imgh, imgw)
        if arg.fixed_size:
            if arg.center_crop:
                # center crop
                j = (imgh - side) // 2
                i = (imgw - side) // 2
                img = img[j:j + side, i:i + side, ...]
            else:
                #random crop
                j = (imgh - side)
                i = (imgw - side)
                h_start = 0
                w_start = 0
                if j != 0:
                    h_start = random.randrange(0, j)
                if i != 0:
                    w_start = random.randrange(0, i)
                img = img[h_start:h_start + side, w_start:w_start + side, ...]
        else:
            if side <= arg.target_size:
                j = (imgh - side)
                i = (imgw - side)
                h_start = 0
                w_start = 0
                if j != 0:
                    h_start = random.randrange(0, j)
                if i != 0:
                    w_start = random.randrange(0, i)
                img = img[h_start:h_start + side, w_start:w_start + side, ...]
            else:
                side = random.randrange(arg.target_size, side)
                j = (imgh - side)
                i = (imgw - side)
                h_start = random.randrange(0, j)
                w_start = random.randrange(0, i)
                img = img[h_start:h_start + side, w_start:w_start + side, ...]
    img = np.array(Image.fromarray(img).resize(size=(arg.target_size, arg.target_size)))
    return img


if __name__ == '__main__':
    img_save_dir = os.path.join(arg.save_dir, arg.dataset_name, arg.mask_type, "imgs")
    masked_save_dir = os.path.join(arg.save_dir, arg.dataset_name, arg.mask_type, "masked_imgs")
    mask_save_dir = os.path.join(arg.save_dir, arg.dataset_name, arg.mask_type, "masks")
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)

    if not os.path.exists(masked_save_dir):
        os.makedirs(masked_save_dir)

    if not os.path.exists(mask_save_dir):
        os.makedirs(mask_save_dir)

    imgs_list = sorted(glob.glob(arg.img_dir + "/*.jpg") + glob.glob(arg.img_dir + "/*.png"))

    if arg.shuffle:
        random.shuffle(imgs_list)
    lama_mask_gen = get_mask_generator(kind=arg.mask_type,max_obj_area=arg.max_obj_area,max_obj_num=arg.max_obj_num)
    total_num = min(len(imgs_list),arg.total_num)
    print("Preparing masks...")
    for i in tqdm(range(total_num)):
        img = resize(np.array(Image.open(imgs_list[i])))
        if "segm" in arg.mask_type:
            mask = lama_mask_gen.get_mask(np.array(img), single_obj=False, objRemoval=True)
        else:
            mask = lama_mask_gen(shape=(arg.target_size, arg.target_size))

        mask = np.expand_dims(mask, axis=-1)
        mask = np.concatenate([mask, mask, mask], axis=-1)
        mask = (mask > 0).astype(np.uint8)
        mask = mask * 255

        masked_img,_ = get_transparent_mask(img,mask)
        Image.fromarray(mask).save(mask_save_dir + f'/{i:0>5d}_im_mask.png')
        Image.fromarray(masked_img).save(masked_save_dir + f'/{i:0>5d}_im_masked.png')
        Image.fromarray(img).save(img_save_dir + f'/{i:0>5d}_im_truth.png')

    print("done.")








