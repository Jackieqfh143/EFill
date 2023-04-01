import cv2
from PIL import Image
import numpy as np
from skimage.feature import canny


def assert_img(input):
    if isinstance(input,str):
        return cv2.imread(input)
    return input


def to_gray(img):
    img = assert_img(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def to_edge(img):
    img = assert_img(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = canny(gray, sigma=2., mask=None).astype(np.float)
    edge = cv2.convertScaleAbs(edge) * 255
    return edge


def to_sobel(img):
    img = assert_img(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(gray, -1, 1, 0, ksize=3, scale=1)
    y = cv2.Sobel(gray, -1, 0, 1, ksize=3, scale=1)
    absx = cv2.convertScaleAbs(x)
    absy = cv2.convertScaleAbs(y)
    sobel_edge = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
    return sobel_edge

#input mask : 0 for background ,1 for missing area
def get_transparent_mask(im_path,mask_path,color=(27,79,114),threshold=0.5):    #(220,20,60)
    im = np.array(Image.open(im_path))
    mask = np.array(Image.open(mask_path)).astype(np.uint8)
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=-1)
    else:
        mask = mask[:, :, 0:1]
    cnts, _ = find_mask_contour(mask)

    threshold = 0.5
    mask = np.where(mask>=threshold,1,0)
    R = np.expand_dims(np.ones(mask.shape[:2]) * color[0],axis=2)
    G = np.expand_dims(np.ones(mask.shape[:2]) * color[1],axis=2)
    B = np.expand_dims(np.ones(mask.shape[:2])* color[2],axis=2)
    color_M = np.concatenate((R,G,B),axis=2)
    color_im = color_M.astype(np.uint8)
    masked_color_im = color_im * mask
    masked_color_im = masked_color_im.astype(np.uint8)

    alpha = 1.0     #img1 transparent ratio
    beta  = 0.5   #img2 transparent ratio
    gamma = 0       #adjust value add to img

    masked_img = cv2.addWeighted(im,alpha,masked_color_im,beta,gamma)
    masked_img = masked_img.astype(np.uint8)
    for c in cnts:
        # draw the contour of the shape on the masked image
        cv2.drawContours(masked_img, [c], -1, (0, 255, 0), 2)

    masked_img = Image.fromarray(masked_img)
    mask = (mask * 255).astype(np.uint8)
    mask = np.concatenate((mask,mask,mask),axis=2)
    mask = Image.fromarray(mask)

    return masked_img,mask

def find_mask_contour(mask):

    mask = (mask > 0).astype(np.uint8)
    # calculate center from mask
    cnts,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    max_cnt = max(cnts,key=cv2.contourArea)

    # compute the center of the contour
    #only return the center of maximum contour
    M = cv2.moments(max_cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cnt_center = (cX, cY)
    return cnts,cnt_center

def draw_bbox(image,bbox,color=(255,0,0),thickness=2):
    start_point, end_point = bbox
    cv2.rectangle(image, start_point, end_point, color, thickness)

def crop_by_bbox(image,bbox):
    tfx,tfy = bbox[0]  #topleft
    drx,dry = bbox[1]  #downright
    croped_img = image[tfy:dry,tfx:drx,:]  # h,w,c
    return croped_img

def resize_and_paste(src_img,obj_img,size=(96,96),to_pos='TopLeft'):
    src_img = Image.fromarray(src_img).resize(size=size)
    merged_im = obj_img.copy()
    if to_pos == 'TopLeft':
        merged_im[:size[1],:size[0],:] = src_img

    return merged_im

def highlight_mask_center(im_path,mask_path,bbox=None,rect_size=(64,64),offset = 10):
    img = np.array(Image.open(im_path))
    mask = np.array(Image.open(mask_path))
    mask = (mask > 0).astype(np.uint8)
    if bbox == None:
        #calculate center from mask
        _,mask_center = find_mask_contour(mask)
        cX,cY = mask_center
        if np.random.binomial(1, 0.5) > 0:
            cX = cX + np.random.randint(0,offset)
        else:
            cX = cX - np.random.randint(0, offset)

        if np.random.binomial(1,0.5) > 0:
            cY = cY + np.random.randint(0, offset)
        else:
            cY = cY - np.random.randint(0, offset)

        topleft_x = cX - rect_size[0] // 2
        topleft_y = cY - rect_size[1] // 2

        downright_x = cX + rect_size[0] // 2
        downright_y = cY + rect_size[0] // 2
        bbox = ((topleft_x,topleft_y),(downright_x,downright_y))

    draw_bbox(img,bbox)
    croped_img = crop_by_bbox(img,bbox)
    try:
        highlighted_im = resize_and_paste(croped_img,img)
    except Exception as e:
        print('Failed to highlight the mask area!')
        return img
    else:
        return highlighted_im







