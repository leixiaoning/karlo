from diffusers import UnCLIPPipeline
import torch
from PIL import Image
import copy
from transformers import CLIPVisionModelWithProjection, CLIPVisionConfig
from diffusers import DiffusionPipeline
import os, cv2
import numpy as np

pipe = DiffusionPipeline.from_pretrained("/www/simple_ssd/lxn3/mtimageblend/plugins/imageblend/models/karlo-v1-alpha-image-variations", \
       torch_dtype=torch.float16, custom_pipeline='karlo/unclip_image_interpolation_lxn.py')
image_encoder = copy.deepcopy(pipe.image_encoder)
feature_extractor = copy.deepcopy(pipe.feature_extractor)
del pipe

pipe = UnCLIPPipeline.from_pretrained("kakaobrain/karlo-v1-alpha", torch_dtype=torch.float16)
image_encoder = image_encoder.to('cuda')
pipe = pipe.to('cuda')

def make_img_emb(img):
    img = Image.open(img).convert("RGB")
    img = feature_extractor(images=img, return_tensors="pt").pixel_values
    img = image_encoder(img.to('cuda', dtype=torch.float16)).image_embeds
    return img

def slerp(val, low, high):
    """
    Find the interpolation point between the 'low' and 'high' values for the given 'val'. See https://en.wikipedia.org/wiki/Slerp for more details on the topic.
    """
    low_norm = low / torch.norm(low)
    high_norm = high / torch.norm(high)
    omega = torch.acos((low_norm * high_norm))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high
    return res

def mix_emb(emb1, emb2, interp_step=0.5):

    temp_image_embeddings = slerp(
                            interp_step, emb1, emb2
                        ).unsqueeze(0)
    return temp_image_embeddings[0]

def pil2cv(img):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if True:


    out_dir = 'output_meiyan/0323/1_rev_model2/'
    os.makedirs(out_dir, exist_ok=True)
    
    l1 = "datatest/meiyan/style0323_1/"
    l2 = "datatest/meiyan/input0323_1/"

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

            prompt = "a high-resolution photograph of a big red frog on a green leaf."
            emb1 = make_img_emb(init_image)
            emb2 = make_img_emb(init_image2)
            
            emb = mix_emb(emb1, emb2, interp_step=0)
                
            a = pipe([prompt],\
                    prepared_image_embedding=emb2).images[0]

            akimg = np.concatenate([cv2_res, pil2cv(a)], axis=1)
            cv2.imwrite(output+'_{}.jpg'.format(0), akimg)
                    
            
            pair_num += 1
    

    

print("done")