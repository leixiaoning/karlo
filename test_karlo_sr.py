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
    dir0 = '/www/simple_ssd/lxn3/diffusers/datas/'
    pair_idx0 = [[0, 3], [0, 2], [0, 5], [6, 4], [8, 1]]
    dir1 = '/www/simple_ssd/lxn3/diffusers/tmp/imgtest/'
    pair_idx1 = [[0, 7], [1, 9], [2, 8], [3, 4], [5, 6]]
    dir2 = '/www/simple_ssd/lxn3/diffusers/tmp/test_2/'
    pair_idx2 = [[0, 21],[11, 15],[24, 3],[3, 25],[12, 4],[19, 14],[26, 21],\
            [11, 18],[11, 6],[11, 5],[3, 9],[29, 7],[27, 26],[1, 18],[8, 9],\
            [25, 8],[8, 24],]
    dirs = [dir0, dir1, dir2]
    pair_idxs = [pair_idx0, pair_idx1, pair_idx2]
    for datai in range(len(dirs)):
        dir = dirs[datai]
        pair_idx = pair_idxs[datai]
        out_dir = "outputs_sr_2/{}_pair_{}_{}"
        #######
        #global __j
        all_imgs = os.listdir(dir)
        for __j in range(len(pair_idx)):
            cv2_res = []
            _i = pair_idx[__j]
            init_image = dir + all_imgs[_i[0]]
            init_image2 = dir + all_imgs[_i[1]]
            output = out_dir.format(datai, _i[0], _i[1])
            output_sr = output.replace('outputs', 'outputs_1024')
            os.makedirs(output.split('/')[0], exist_ok=True)
            os.makedirs(output_sr.split('/')[0], exist_ok=True)
            img_s = Image.open(init_image).convert("RGB")
            cv2_res.append(pil2cv(img_s.resize([256,256])))
            img_e = Image.open(init_image2).convert("RGB")
            cv2_res.append(pil2cv(img_e.resize([256,256])))
            cv2_res = np.concatenate(cv2_res, axis=1)
            a  = unclip_image_interpolation(img_s, img_e, 10, 2023)
            for k in range(len(a)):
                akimg = np.concatenate([cv2_res, pil2cv(a[k])], axis=1)
                cv2.imwrite(output+'_{}.jpg'.format(k), akimg)

                # sr
                b = sd_karlo(a[k])
                cv2.imwrite(output_sr+'_{}.jpg'.format(k), pil2cv(b[0]))

print("done")