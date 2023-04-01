import glob
import os
from .html import HTML
import random




def show_im2_html(web_title,web_header,web_dir,img_dir,im_size=256,max_num = 200):
    webpage = HTML(web_dir, web_title)
    webpage.add_header(web_header)

    imgs = sorted(glob.glob(img_dir + "/*.png") + glob.glob(img_dir + "/*.jpg"))

    assert max_num <= len(imgs)

    selected_imgs_idx = [random.randint(0,len(imgs) // 3 - 1) for i in range(max_num)]

    for idx in selected_imgs_idx:
        masked_name = f'{idx:0>5d}_im_masked.png'
        comp_name = f'{idx:0>5d}_im_out.png'
        gt_name = f'{idx:0>5d}_im_truth.png'
        txts = [masked_name,comp_name,gt_name]
        ims = []
        links = []
        for n in txts:
            im_path = os.path.abspath(os.path.join(img_dir, n))
            ims.append(im_path)
            links.append(im_path)

        webpage.add_images(ims, txts, links, width=im_size)

    webpage.save()


if __name__ == '__main__':
    show_im2_html(web_title=f"Result_place",
                  web_header=f"Inpainting Results on place",
                  web_dir="./",
                  img_dir="/home/codeoops/CV/Remote4/results/Place/thick_256",
                  im_size=256,
                  max_num=200)

