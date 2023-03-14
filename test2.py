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

pipe = DiffusionPipeline.from_pretrained("kakaobrain/karlo-v1-alpha-image-variations", torch_dtype=dtype, custom_pipeline='unclip_image_interpolation')
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
    out_dir = "outputs_test2/"
    os.makedirs(out_dir, exist_ok=True)
    testset = '/www/simple_ssd/lxn3/diffusers/test03/test0314'        
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
        

print("done")
