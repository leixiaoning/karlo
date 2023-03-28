from diffusers import DiffusionPipeline
import gradio as gr
import torch
import math, os
from PIL import Image
import cv2
import numpy as np
from scripts.animal_detect import AnimalDetect

animal_process = AnimalDetect('/www/simple_ssd/lxn3/tools/mtanimal-1.3.25/models')

def pil2cv(img):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def read_img(imgpath, animal):
    if animal:
        img = animal_process.get_crop(imgpath)
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    else:
        img = Image.open(imgpath).convert("RGB")
    return img

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.bfloat16

pipe = DiffusionPipeline.from_pretrained("/www/simple_ssd/lxn3/mtimageblend/plugins/imageblend/models/karlo-v1-alpha-image-variations", \
       torch_dtype=dtype, custom_pipeline='karlo/unclip_image_interpolation_lxn.py')
pipe.to(device)

def unclip_image_interpolation(
  start_image,
  end_image,
  steps,
  seed,
  blend_weight=None
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    images = [start_image, end_image]
    output = pipe(image=images, steps=steps, \
            decoder_num_inference_steps=25, \
            super_res_num_inference_steps=7,\
            generator=generator,\
            blend_weight=blend_weight)
    return output.images


if True:
    out_dir = 'output_meiyan/0323/animal_1/'
    os.makedirs(out_dir, exist_ok=True)
    blend_weight=[0.4, 0.5, 0.6] #[1-__i for __i in [0.25,0.3,0.35]]
    l1 = "datatest/meiyan/dogcat/style/"
    l2 = "datatest/meiyan/dogcat/cat/"
    animal = True

    pair_num = 0
    for i1 in os.listdir(l1):
        for i2 in os.listdir(l2):
            cv2_res = []
            init_image = os.path.join(l1, i1)
            img_s = read_img(init_image, False)
            cv2_res.append(pil2cv(img_s.resize([256,256])))
            init_image2 = os.path.join(l2, i2)
            img_e = read_img(init_image2, animal)
            cv2_res.append(pil2cv(img_e.resize([256,256])))
            cv2_res = np.concatenate(cv2_res, axis=1)

            output = out_dir+'{}'.format(pair_num)

            a  = unclip_image_interpolation(img_s, img_e, len(blend_weight), 2023, blend_weight=blend_weight)
            for k in range(len(a)):
                akimg = np.concatenate([cv2_res, pil2cv(a[k])], axis=1)
                cv2.imwrite(output+'_{}.jpg'.format(k), akimg)
            
            pair_num += 1

print("done")
