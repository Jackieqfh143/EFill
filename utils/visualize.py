import glob
import os
from .html import HTML
import random
from .util import get_file_info




def show_im2_html(web_title,web_header,web_dir,img_dir,im_size=256,max_num = 200):
    webpage = HTML(web_dir, web_title)
    webpage.add_header(web_header)

    imgs = sorted(glob.glob(img_dir + "/*.png") + glob.glob(img_dir + "/*.jpg"))

    assert max_num <= len(imgs)

    selected_imgs_idx = [random.randint(0,len(imgs) // 3 - 1) for i in range(max_num)]

    img_suffex = get_file_info(imgs[1])["file_type"]

    for idx in selected_imgs_idx:
        masked_name = f'{idx:0>5d}_im_masked{img_suffex}'
        comp_name = f'{idx:0>5d}_im_out{img_suffex}'
        gt_name = f'{idx:0>5d}_im_truth{img_suffex}'
        txts = [masked_name,comp_name,gt_name]
        ims = []
        links = []
        for n in txts:
            im_path = os.path.abspath(os.path.join(img_dir, n))
            ims.append(im_path)
            links.append(im_path)

        webpage.add_images(ims, txts, links, width=im_size)

    webpage.save()



