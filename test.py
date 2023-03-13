"""
from diffusers import UnCLIPPipeline
import torch

pipe = UnCLIPPipeline.from_pretrained("kakaobrain/karlo-v1-alpha", torch_dtype=torch.float16)
pipe = pipe.to('cuda')

prompt = "a high-resolution photograph of a big red frog on a green leaf."
image = pipe(prompt).images[0]
image.save("./frog.png")
"""

###########


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
"""
examples = [
  ["starry_night.jpg","dogs.jpg", 10, 20],
  ["flowers.jpg", "dogs.jpg", 10, 42],
  ["starry_night.jpg","flowers.jpg", 10, 9011]
]

n=0
for _s, _e, _step, _seed in examples:
    img_s = Image.open(_s).convert("RGB")
    img_e = Image.open(_e).convert("RGB")
    a  = unclip_image_interpolation(img_s, img_e, _step, _seed)
    for k in range(len(a)):
        a[k].save('outputs/{}_{}.jpg'.format(n, k))
    n+=1

print("done")
"""
import os
if True:
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
        out_dir = "outputs1/{}_pair_{}_{}"
        #######
        global __j
        all_imgs = os.listdir(dir)
        for __j in range(len(pair_idx)):
            cv2_res = []
            _i = pair_idx[__j]
            init_image = dir + all_imgs[_i[0]]
            init_image2 = dir + all_imgs[_i[1]]
            output = out_dir.format(datai, _i[0], _i[1])
            os.makedirs(output.split('/')[0], exist_ok=True)
            img_s = Image.open(init_image).convert("RGB")
            cv2_res.append(pil2cv(img_s.resize([256,256])))
            img_e = Image.open(init_image2).convert("RGB")
            cv2_res.append(pil2cv(img_e.resize([256,256])))
            cv2_res = np.concatenate(cv2_res, axis=1)
            a  = unclip_image_interpolation(img_s, img_e, 10, 2023)
            for k in range(len(a)):
                akimg = np.concatenate([cv2_res, pil2cv(a[k])], axis=1)
                cv2.imwrite(output+'_{}.jpg'.format(k), akimg)
print("done")
"""
inputs = [
  gr.Image(type="pil"),
  gr.Image(type="pil"),
  gr.Slider(minimum=2, maximum=12, default=5, step=1, label="Steps"),
  gr.Number(0, label="Seed", precision=0)
]

output = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        ).style(grid=[2], height="auto")

examples = [
  ["starry_night.jpg","dogs.jpg", 5, 20],
  ["flowers.jpg", "dogs.jpg", 5, 42],
  ["starry_night.jpg","flowers.jpg", 6, 9011]
]

title = "UnClip Image Interpolation Pipeline"

demo_app = gr.Interface(
    fn=unclip_image_interpolation,
    inputs=inputs,
    outputs=output,
    title=title,
    theme='huggingface',
    examples=examples,
    cache_examples=True
)
demo_app.launch(debug=True, enable_queue=True)
"""