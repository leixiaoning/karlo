from diffusers import DiffusionPipeline
import gradio as gr
import torch
import math
from PIL import Image
import cv2
import numpy as np
def pil2cv(img):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.bfloat16

pipe = DiffusionPipeline.from_pretrained("kakaobrain/karlo-v1-alpha-image-variations", \
       torch_dtype=dtype, custom_pipeline='karlo/unclip_image_interpolation_lxn.py')
pipe.to(device)

def unclip_image_interpolation(
  start_image,
  end_image,
  steps,
  seed
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    images = [start_image, end_image]
    output = pipe(image=images, steps=steps, \
            decoder_num_inference_steps=25, \
            super_res_num_inference_steps=7,\
            generator=generator)
    return output.images

import os
if True:
    """
    out_dir = "outputs_test316/"
    os.makedirs(out_dir, exist_ok=True)
    testset = '/www/simple_ssd/lxn3/diffusers/test03/test0315'        
    sub1 = os.listdir(testset)
    for sub in sub1:
        sub2 = os.path.join(testset, sub)
        sub3 = os.listdir(sub2)
        cv2_res = []
        init_image = os.path.join(sub2, sub3[0])
        img_s = Image.open(init_image).convert("RGB")
        cv2_res.append(pil2cv(img_s.resize([256,256])))
        init_image2 = os.path.join(sub2, sub3[1])
        img_e = Image.open(init_image2).convert("RGB")
        cv2_res.append(pil2cv(img_e.resize([256,256])))
        cv2_res = np.concatenate(cv2_res, axis=1)

        output = out_dir+'{}'.format(sub)

        a  = unclip_image_interpolation(img_s, img_e, 10, 2023)
        for k in range(len(a)):
            akimg = np.concatenate([cv2_res, pil2cv(a[k])], axis=1)
            cv2.imwrite(output+'_{}.jpg'.format(k), akimg)
    """    

    out_dir = 'outputs_test_meiyan_9/'
    os.makedirs(out_dir, exist_ok=True)
    #testset = 'datatest/320_3'
    #l1 = os.path.join(testset, 'input')
    #l2 = os.path.join(testset, 'style')
    l1 = "datatest/meiyan/blendtest2/input/"
    l2 = "datatest/meiyan/blendtest2/style/_9_年龄变化/"

    pair_num = 0
    for i1 in os.listdir(l1):
        for i2 in os.listdir(l2):
            cv2_res = []
            init_image = os.path.join(l1, i1)
            img_s = Image.open(init_image).convert("RGB")
            cv2_res.append(pil2cv(img_s.resize([256,256])))
            init_image2 = os.path.join(l2, i2)
            img_e = Image.open(init_image2).convert("RGB")
            cv2_res.append(pil2cv(img_e.resize([256,256])))
            cv2_res = np.concatenate(cv2_res, axis=1)

            output = out_dir+'{}'.format(pair_num)

            a  = unclip_image_interpolation(img_s, img_e, 3, 2023)
            for k in range(len(a)):
                akimg = np.concatenate([cv2_res, pil2cv(a[k])], axis=1)
                cv2.imwrite(output+'_{}.jpg'.format(k), akimg)
            
            pair_num += 1

print("done")
