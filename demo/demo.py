import glob
import os
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, url_for, render_template, request, redirect, send_from_directory
from PIL import Image
from model import *
import base64
import io
import random
import time
import torch
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--port',type=int,dest='port',default=8000,help='')
parse.add_argument('--model_path',type=str,dest='port',default="../checkpoints/place_best.pth",help='path to model checkpoints')
parse.add_argument('--config_path',type=str,dest='port',default="../config/configs.yaml",help='path to model config')
opt = parse.parse_args()


max_size = 512
max_num_examples = 50
UPLOAD_FOLDER = 'static/images'
filename = ""
origin_imgName = ""
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'jpeg', 'bmp'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

port = opt.port

def load_img(path):
    image = Image.open(path)
    W, H = image.size
    if max(W, H) > max_size:
        ratio = float(max_size) / max(W, H)
        W = int(W * ratio)
        H = int(H * ratio)
        image = image.resize((W, H))

    return image

def process_image(img, mask, name, opt, save_to_input=True):
    img =img.convert("RGB")
    w_raw, h_raw = img.size
    h_t, w_t = h_raw//8*8, w_raw//8*8
    img = np.array(img.resize((w_t, h_t)))
    mask = np.array(mask.resize((w_t, h_t)))
    if len(mask.shape) ==2:
        mask = np.expand_dims(mask, axis=-1)
        mask = np.concatenate((mask, mask, mask), axis=-1)
    mask = mask.astype(np.uint8)
    model = InpaintingModel(model_path=opt.model_path, config_path=opt.config_path)
    result = model(img,mask)  #inpainting process

    result = Image.fromarray(result).resize((w_raw, h_raw))
    result = np.array(result)
    result = Image.fromarray(result.astype(np.uint8))
    result.save(f"static/results/{name}")
    if save_to_input:
        global filename
        filename = str(time.time()) + '.jpg'
        result.save(f"static/images/{filename}")

@app.route('/', methods=['GET', 'POST'])
def hello(name=None):
    global filename,origin_imgName
    if 'example' in request.form:
        filename= request.form['example']
        origin_imgName = filename
        image = load_img(os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
        W, H = image.size
        return render_template('Inpainting.html', name=name, origin_imgName = origin_imgName,image_name=filename, image_width=W,
                image_height=H,list_examples=list_examples)
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                image = load_img(file)
                W, H = image.size
                filename = "resize_"+filename
                origin_imgName = "resize_"+filename
                image.save(os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
                return render_template('Inpainting.html', name=name, origin_imgName = origin_imgName, image_name=filename, image_width=W,
                        image_height=H,list_examples=list_examples)
            else:
                filename=list_examples[0]
                image = load_img(os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
                W, H = image.size
                return render_template('Inpainting.html', name=name, origin_imgName = origin_imgName,image_name=filename, image_width=W, image_height=H,
                        is_alert=True,list_examples=list_examples)

        if 'mask' in request.form:
            # filename = request.form['imgname']
            mask_data = request.form['mask']
            mask_data = mask_data.replace('data:image/png;base64,', '')
            mask_data = mask_data.replace(' ', '+')
            mask = base64.b64decode(mask_data)
            maskname = '.'.join(filename.split('.')[:-1]) + '.png'
            maskname = maskname.replace("/","_")
            maskname = "{}_{}".format(int(time.time()), maskname)
            with open(os.path.join('static/masks', maskname), "wb") as fh:
                fh.write(mask)
            mask = io.BytesIO(mask)
            mask = Image.open(mask).convert("L")
            image = Image.open(f"static/images/{filename}")
            W, H = image.size
            list_op = ["result"]
            for op in list_op:
                process_image(image, mask, f"{op}_"+maskname, op, save_to_input=True)
            return render_template('Inpainting.html', name=name, origin_imgName = origin_imgName,image_name=filename,
                    mask_name=maskname, image_width=W, image_height=H, list_opt=list_op,list_examples=list_examples)
    else:
        filename = list_examples[0]
        origin_imgName = filename
        image = load_img(os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
        W, H = image.size
        return render_template('Inpainting.html', name=name, origin_imgName = origin_imgName,image_name=filename, image_width=W, image_height=H,
                list_examples=list_examples)



if __name__ == "__main__":
    list_examples = sorted(glob.glob(UPLOAD_FOLDER + '/*.jpg'))
    list_examples = [item.replace(UPLOAD_FOLDER+'/', '') for item in list_examples]
    app.run(host='0.0.0.0', debug=True, port=port, threaded=True) #run with flask
