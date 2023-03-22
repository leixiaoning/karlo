from models.generate import sdsr
from diffusers import DiffusionPipeline
import torch
from PIL import Image
import cv2
import numpy as np
import os


if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.bfloat16

pipe = DiffusionPipeline.from_pretrained("kakaobrain/karlo-v1-alpha-image-variations", torch_dtype=dtype, custom_pipeline='unclip_image_interpolation')
pipe.to(device)

def pil2cv(img):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

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

sr = sdsr(scheduler='DDIM', cpu=False, xfm=True)
def sd_karlo(images=None):
    if images == None:
        images = sr.generate(
                    cpu=False,
                    prompt='a painting of a polar bear drinking hot chocolate',#prompt,
                    n_images=1,
                    n_prior=25,
                    n_decoder=25,
                    n_super_res=7,
                    cfg_prior=4.0,
                    cfg_decoder=4.0,
                )
    else:
        if not isinstance(images, list):
            images=[images]

    images_up = sr.upscale(
                            cpu=False,
                            xfm=True,#xfm_sd
                            downscale=256,
                            scheduler='DDIM',
                            prompt='',
                            neg_prompt='',
                            images=images,
                            n_steps=50,
                            cfg=7.5,
                        )
    #images[0].save('tmp1.jpg')
    #images_up[0].save('tmp2.jpg')      
    #print("done")
    return images_up


if __name__ == "__main__":
    input_dir = 'outputs_test320_3/'
    output_dir = 'outputs_test320_3_sr/'
    os.makedirs(output_dir, exist_ok=True)
    l = os.listdir(input_dir)
    for i in range(len(l)):
        imgpath = os.path.join(input_dir, l[i])
        if not os.path.exists(imgpath.replace(input_dir, output_dir)):
            try:
                img = Image.open(imgpath).convert("RGB")
                imgsr = []
                for n in range(3):
                    imgi = img.crop((256*n,0,256*(1+n),256))
                    imgi = sd_karlo(imgi)
                    imgsr.append(pil2cv(imgi[0]))
                imgsr = np.concatenate(imgsr, axis=1)
                cv2.imwrite(imgpath.replace(input_dir, output_dir), imgsr)
            except:
                print('failed: ')
                print(imgpath)