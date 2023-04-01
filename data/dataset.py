import os
import math
import torch
from torch.utils import data
import random
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as transform
import cv2
from skimage.io import imread
from skimage.feature import canny
import traceback
from PIL import Image,ImageDraw
import albumentations as A
from data.Lama_mask_gen import RandomSegmentationMaskGenerator,get_mask_generator

class Dataset(data.Dataset):
    def __init__(self, image_path, mask_path, mask_mode, target_size, augment=True, training=True,
                 mask_reverse=True, center_crop=False, data_type='resize', min_mask_ratio=5, max_mask_ratio=70,
                 get_edge=False,rect_size=64,edge_type='sobel',seed=2022,
                 get_seg_mask=False):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.center_crop = center_crop
        self.data = self.load_list(image_path)
        self.mask_data = self.load_list(mask_path)

        if get_seg_mask:
            self.seg_mask_generator = RandomSegmentationMaskGenerator()
        else:
            self.seg_mask_generator = None

        self.mask_seed = seed
        self.mask_generator = RandomMask(s=target_size,hole_range=[min_mask_ratio/100,max_mask_ratio/100])
        self.lama_mask_gen = get_mask_generator(kind='default')
        self.rect_size = rect_size
        self.data_type = data_type
        self.edge_type = edge_type
        self.target_size = target_size
        self.mask_type = mask_mode
        self.mask_reverse = mask_reverse
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.get_edge = get_edge
        if self.augment:
            self.img_augment = A.Compose([
                A.HorizontalFlip(),
                A.OpticalDistortion(),
                A.CLAHE()])

        # for external mask only
        self.mask_transform = transform.Compose([
            transform.RandomVerticalFlip(0.5),
            transform.RandomHorizontalFlip(0.5),
            transform.Resize(self.target_size),
        ])

        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            traceback.print_exc()
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_item(self, index):
        img = cv2.imread(self.data[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.training:
            resize_args = {"aspect_ratio_kept":True, "fixed_size":False, "centerCrop":False}

            if self.center_crop:
                img = self.resize(img, True, True, True)
            else:
                img = self.resize(img, **resize_args)

            if self.augment:
                img = self.img_augment(image=img)['image']

        else:
            resize_args = {"aspect_ratio_kept": True, "fixed_size": True, "centerCrop": True}
            img = self.resize(img, **resize_args)

        # load mask
        if self.training:
            mask_type = np.random.choice([0, 1], 1, p=[0.1, 0.9])
            mask = self.load_mask(index, mask_type, img=img)
        else:
            mask = self.load_mask(index, self.mask_type)

        if self.get_edge:
            # load edge
            gray_im = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            if self.edge_type == 'canny':
                edge = self.load_edge(gray_im)          #canny edge
            else:
                edge = load_sobel_edge(gray_im)        #sobel edge
            edge = self.to_tensor(edge)
        else:
            edge = self.to_tensor(np.ones_like(img))

        img = self.to_tensor(img)
        mask = self.to_tensor(mask)

        # augment external mask
        if self.mask_type == 0:
            mask = self.mask_transform(mask)

        return [img,mask,edge]

    def load_mask(self, index,mask_type,img=None):
        # external mask, random order
        if mask_type == 0:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            mask = self.resize(mask, False)
            mask = (mask > 0).astype(np.uint8)  # threshold due to interpolation
            mask = np.expand_dims(mask, axis=2)
            mask = np.concatenate([mask, mask, mask], axis=2)
            if self.mask_reverse:
                return (1 - mask) * 255
            else:
                return mask * 255

        # generate random mask
        if mask_type == 1:
            mask = get_random_mask(im_size=self.target_size,mask_size=self.rect_size,seed=self.mask_seed,img = img,
                                   mask_gen=self.mask_generator,seg_mask_gen=self.seg_mask_generator,lama_mask_gen=self.lama_mask_gen,
                                   target_size=self.target_size)
            # mask = self.resize(mask, False)
            mask = (mask > 0).astype(np.uint8)
            if self.mask_reverse:
                return (1 - mask) * 255
            else:
                return mask * 255

        # generate random square mask
        if mask_type == 2:
            if self.training:
                mask, _ = generate_rect_mask(self.target_size, self.rect_size)
            else:
                mask, _ = generate_rect_mask(self.target_size, self.rect_size, rand_mask=False)  # central square for testing mode

            mask = (mask > 0).astype(np.uint8)

            if self.mask_reverse:
                return (1 - mask) * 255
            else:
                return mask * 255

        # external mask, fixed order
        if mask_type == 3:
            mask_index = index
            mask = imread(self.mask_data[mask_index])
            mask = self.resize(mask, False)
            mask = (mask > 0).astype(np.uint8)  # threshold due to interpolation
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=2)
                mask = np.concatenate([mask, mask, mask], axis=2)

            if self.mask_reverse:
                return (1 - mask) * 255
            else:
                return mask * 255

    def resize(self, img, aspect_ratio_kept=True, fixed_size=False, centerCrop=False):

        if aspect_ratio_kept:
            imgh, imgw = img.shape[0:2]
            side = np.minimum(imgh, imgw)
            if fixed_size:
                if centerCrop:
                    # center crop
                    j = (imgh - side) // 2
                    i = (imgw - side) // 2
                    img = img[j:j + side, i:i + side, ...]
                else:
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
                if side <= self.target_size:
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
                    side = random.randrange(self.target_size, side)
                    j = (imgh - side)
                    i = (imgw - side)
                    h_start = random.randrange(0, j)
                    w_start = random.randrange(0, i)
                    img = img[h_start:h_start + side, w_start:w_start + side, ...]
        img = np.array(Image.fromarray(img).resize(size=(self.target_size, self.target_size)))
        return img

    def to_tensor(self, img, toRGB=False):
        img = Image.fromarray(img)
        if toRGB:
            img = img.convert('RGB')
        img_t = F.to_tensor(img).float()      #normalize to 0 ~ 1
        return img_t

    def load_list(self, path):
        if isinstance(path, str):
            if os.path.isdir(path):
                files_list = self.getfilelist(path)
                files_list.sort()
                return files_list

            if os.path.isfile(path):
                try:
                    files_list = list(np.genfromtxt(path, dtype=np.str, encoding='utf-8'))
                    files_list.sort()
                    return files_list

                except:
                    print('Failed to extract data from txt files...')
                    return [path]

        elif isinstance(path, list):
            out_path = []
            for p in path:
                if os.path.isdir(p):
                    p_ = self.getfilelist(p)
                    p_.sort()
                    out_path += p_
                if os.path.isfile(p):
                    try:
                        out_path += np.genfromtxt(p, dtype=np.str, encoding='utf-8')
                    except:
                        out_path += [p]

            return out_path
        return []

    def load_edge(self, img):
        return canny(img, sigma=2.).astype(np.float)

    def getfilelist(self, path):
        all_file=[]
        for dir,folder,file in os.walk(path):
            for i in file:
                t = "%s/%s"%(dir,i)
                if t.endswith('.png') or t.endswith('.jpg') or t.endswith('.JPG') or t.endswith('.PNG') or t.endswith('.JPEG'):
                    all_file.append(t)

        return all_file

def get_random_mask(im_size,mask_size=64,seed=2022,img = None,mask_gen=None,seg_mask_gen=None,lama_mask_gen=None,
                    target_size=256):

    choice = np.random.choice([0,1],1,p=[0.8,0.2])

    if choice == 0:
        #lama large mask
        if lama_mask_gen == None:
            lama_mask_gen = get_mask_generator(kind='default')
        mask = lama_mask_gen(shape=(target_size,target_size))
        mask = np.expand_dims(mask, axis=-1)
        mask = np.concatenate([mask, mask, mask], axis=-1)

    # if choice == 0:
    #     # print('Comod mask')
    #     #comodgan mask
    #     if mask_gen == None:
    #         mask_gen = RandomMask(s=target_size)
    #         seed = np.random.randint(0, 999999)
    #     mask = mask_gen(seed=seed)
    #     mask = np.expand_dims(mask, axis=2)
    #     mask = np.concatenate([mask, mask, mask], axis=2)

    # elif choice == 1:
    #     #LaMa segmentaion mask
    #     if seg_mask_gen != None:
    #         mask = seg_mask_gen(img)
    #         mask = np.expand_dims(mask, axis=-1)
    #         mask = np.concatenate([mask, mask, mask], axis=-1)
    #         mask = (mask * 255).astype(np.uint8)
    #         mask = np.array(Image.fromarray(mask).resize(target_size,target_size))
    #     else:
    #

    else:
        # print('deepfill random rect mask')
        #deepfill random rect mask
        mask = np.zeros((im_size, im_size,3)).astype(np.float32)
        if np.random.binomial(1, 0.5) > 0:
            mask_rec_num = np.random.randint(1,5)
            for i in range(mask_rec_num):
                rec_mask,_ = generate_rect_mask(im_size=im_size,mask_size=mask_size)
                mask += rec_mask
        else:
            #fixed center square mask
            mask_size_ = min(im_size // 2, 2 * mask_size)
            rec_mask, _ = generate_rect_mask(im_size=im_size, mask_size=mask_size_,rand_mask=False)
            mask += rec_mask

    return mask

#DeepFill mask
def generate_stroke_mask(im_size, max_parts=7, maxVertex=20, maxLength=60, maxBrushWidth=40, maxAngle=360):
    mask = np.zeros((im_size, im_size, 1), dtype=np.float32)
    parts = random.randint(1, max_parts)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size, im_size)
    mask = np.minimum(mask, 1.0)
    mask = np.concatenate([mask, mask, mask], axis=2)
    return mask


def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(20, h - 20)
    startX = np.random.randint(20, w - 20)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)
        nextY = np.maximum(np.minimum(nextY, h - 20), 20).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 20), 20).astype(np.int)
        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        # if np.random.binomial(1, 0.5) > 0:
        #     cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        startY, startX = nextY, nextX
        if startX <= 5:
            startX += 15
        if startY <= 5:
            startY += 15
    # cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask


def generate_rect_mask(im_size, mask_size, margin=8, rand_mask=True):
    mask = np.zeros((im_size, im_size)).astype(np.float32)
    if rand_mask:
        sz0, sz1 = mask_size, mask_size
        of0 = np.random.randint(margin, im_size - sz0 - margin)
        of1 = np.random.randint(margin, im_size - sz1 - margin)
    else:
        sz0, sz1 = mask_size, mask_size
        of0 = (im_size - sz0) // 2
        of1 = (im_size - sz1) // 2
    mask[of0:of0 + sz0, of1:of1 + sz1] = 1
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    rect = np.array([[of0, sz0, of1, sz1]], dtype=int)
    return mask, rect

def load_sobel_edge(gray):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(gray, -1, 1, 0, ksize=3, scale=1)
    y = cv2.Sobel(gray, -1, 0, 1, ksize=3, scale=1)
    absx = cv2.convertScaleAbs(x)
    absy = cv2.convertScaleAbs(y)
    edge = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)

    return edge

#CoModGAN mask
class RandomMask:
    def __init__(self, s, hole_range=[0.05,0.6]):
        self.s = s
        self.hole_range = hole_range
        self.rng_seed_train = np.random.RandomState()


    def RandomBrush(
        self,
        max_tries,
        s,
        min_num_vertex = 4,
        max_num_vertex = 18,
        mean_angle = 2*math.pi / 5,
        angle_range = 2*math.pi / 15,
        min_width = 12,
        max_width = 48):
        H, W = s, s
        average_radius = math.sqrt(H*H+W*W) / 8
        mask = Image.new('L', (W, H), 0)
        for _ in range(self.rng_seed_train.randint(max_tries)):
            num_vertex = self.rng_seed_train.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - self.rng_seed_train.uniform(0, angle_range)
            angle_max = mean_angle + self.rng_seed_train.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2*math.pi - self.rng_seed_train.uniform(angle_min, angle_max))
                else:
                    angles.append(self.rng_seed_train.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(self.rng_seed_train.randint(0, w)), int(self.rng_seed_train.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    self.rng_seed_train.normal(loc=average_radius, scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(self.rng_seed_train.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                            v[1] - width//2,
                            v[0] + width//2,
                            v[1] + width//2),
                            fill=1)
            if self.rng_seed_train.random() > 0.5:
                mask.transpose(Image.FLIP_LEFT_RIGHT)
            if self.rng_seed_train.random() > 0.5:
                mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.uint8)
        if self.rng_seed_train.random() > 0.5:
            mask = np.flip(mask, 0)
        if self.rng_seed_train.random() > 0.5:
            mask = np.flip(mask, 1)
        return mask

    def Fill(self, mask, max_size):
        w, h = self.rng_seed_train.randint(max_size), self.rng_seed_train.randint(max_size)
        ww, hh = w // 2, h // 2
        x, y = self.rng_seed_train.randint(-ww, self.s - w + ww), self.rng_seed_train.randint(-hh, self.s - h + hh)
        mask[max(y, 0): min(y + h, self.s), max(x, 0): min(x + w, self.s)] = 0

    def MultiFill(self, mask, max_tries, max_size):
        for _ in range(self.rng_seed_train.randint(max_tries)):
            self.Fill(mask, max_size)

    def __call__(self, seed):
        if seed is not None:
            # print(f'fixed seed {seed}')
            self.rng_seed_train = np.random.RandomState(seed)

        mask = np.ones((self.s, self.s), np.uint8)
        while True:
            coef = min(self.hole_range[0] + self.hole_range[1], 1.0)
            self.MultiFill(mask, int(10 * coef), self.s // 2)
            self.MultiFill(mask, int(5 * coef), self.s)
            mask = np.logical_and(mask, 1 - self.RandomBrush(int(20 * coef), self.s))
            hole_ratio = 1 - np.mean(mask)
            if self.hole_range is not None and (hole_ratio <= self.hole_range[0] or hole_ratio >= self.hole_range[1]):
                mask.fill(1)
                continue
            else:
                break
        mask = mask.astype(np.float32)
        mask = np.clip(mask, 0, 1.0)
        mask = 1 - mask         # 1 for holes
        # mask = (mask * 255).astype(np.uint8)
        return mask



    def call_rectangle(self, seed):
        if seed is not None:
            # print(f'fixed seed {seed}')
            self.rng_seed_train = np.random.RandomState(seed)

        mask = np.ones((self.s, self.s), np.uint8)
        while True:
            coef = min(self.hole_range[0] + self.hole_range[1], 1.0)
            self.MultiFill(mask, int(10 * coef), self.s // 2)
            self.MultiFill(mask, int(5 * coef), self.s)
            # mask = np.logical_and(mask, 1 - self.RandomBrush(int(20 * coef), self.s))
            hole_ratio = 1 - np.mean(mask)
            if self.hole_range is not None and (hole_ratio <= self.hole_range[0] or hole_ratio >= self.hole_range[1]):
                mask.fill(1)
                continue
            else:
                break
        mask = mask.astype(np.float32)
        mask = np.clip(mask, 0, 1.0)
        mask = 1 - mask     # 1 for holes
        # mask = (mask * 255).astype(np.uint8)
        return mask



if __name__ == '__main__':
    data_dir = '/home/codeoops/CV/data/place_512_raw/data_large'
    save_file = '/home/codeoops/CV/data/place_512_raw/train_large_test.flist'
    test_img = '/home/codeoops/CV/data/Celeba-hq/test_256/3.jpg'
    img_aug = A.Compose([
                A.OpticalDistortion(p=1)])
    img = imread(test_img)
    auged_img = img_aug(image=img)['image']
    Image.fromarray(auged_img).show()
    gray_img_ = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_img = np.array(Image.fromarray(img).convert('L'))
    print()





