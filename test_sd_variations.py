from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from torchvision import transforms
import os
import cv2
import numpy as np
import torch

device = "cuda:0"
sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
  "/www/simple_ssd/lxn3/data_download/sd-image-variations-diffusers",
  revision="v2.0",
  )
sd_pipe = sd_pipe.to(device)

tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),
        transforms.Normalize(
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711]),
    ])

def pil2cv(img):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def sdmix(util, img1, img2):
    #sd pipeline of image variation 
    imgs = sd_pipe(torch.cat([img1, img2]), guidance_scale=7.5, mixemb=True)
    #emb = util._encode_image(torch.cat([img1, img2]))
    
    return imgs

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
        out_dir = "outputs_var2/{}_pair_{}_{}"

        all_imgs = os.listdir(dir)
        for __j in range(len(pair_idx)):
            cv2_res = []
            _i = pair_idx[__j]
            init_image = dir + all_imgs[_i[0]]
            init_image2 = dir + all_imgs[_i[1]]
            output = out_dir.format(datai, _i[0], _i[1])
            os.makedirs(output.split('/')[0], exist_ok=True)
            img_s = Image.open(init_image).convert("RGB")
            cv2_res.append(pil2cv(img_s.resize([512,512])))
            img_e = Image.open(init_image2).convert("RGB")
            cv2_res.append(pil2cv(img_e.resize([512,512])))
            cv2_res = np.concatenate(cv2_res, axis=1)

            im = Image.open(init_image)
            inp = tform(im).to(device).unsqueeze(0)
            im2 = Image.open(init_image2)
            inp2 = tform(im2).to(device).unsqueeze(0)

            out = sdmix(sd_pipe, inp, inp2)["images"]
            for k in range(len(out)):
                akimg = np.concatenate([cv2_res, pil2cv(out[k])], axis=1)
                cv2.imwrite(output+'_{}.jpg'.format(k), akimg)
            #out = sd_pipe(torch.cat([inp, inp2]), guidance_scale=3)
            #out["images"][0].save("result.jpg")

    print("done")