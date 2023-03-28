import types
from typing import Union, List, Optional, Callable
import diffusers
from diffusers import DiffusionPipeline
import gradio as gr
import torch
import math
from PIL import Image
import cv2
import numpy as np
from useblip import Img2TextTool

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.bfloat16

## img2text blip2
i2t = Img2TextTool(model_root='/www/simple_ssd/data/model_ckpt/')

def get_txt(imgpath):
    #blip 自动获取文本
    img = Image.open(imgpath).convert("RGB")
    img = img.resize((512, 512))
    t = i2t.img2text(img) 
    return t[0]
    
def pil2cv(img):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)



## 扩散模型 加载
pipe = DiffusionPipeline.from_pretrained("kakaobrain/karlo-v1-alpha-image-variations", \
       torch_dtype=dtype, custom_pipeline='karlo/unclip_image_interpolation_lxn.py')
pipe.to(device)

def unclip_image_interpolation(
  start_image,
  end_image,
  steps,
  seed,
  image_embeddings=None
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    images = [start_image, end_image]
    output = pipe(image=images, steps=steps, \
            decoder_num_inference_steps=25, \
            super_res_num_inference_steps=7,\
            generator=generator,\
            text_prior_emb=image_embeddings)
    return output.images

@torch.inference_mode()
def karlo_prior(
    self,
    prompt: Union[str, List[str]],
    num_images_per_prompt: int = 1,
    prior_num_inference_steps: int = 25,
    generator: Optional[torch.Generator] = None,
    prior_latents: Optional[torch.FloatTensor] = None,
    prior_guidance_scale: float = 4.0,
):
    """
    copy from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/unclip/pipeline_unclip.py#L234-L358
    """
    if isinstance(prompt, str):
        batch_size = 1
    elif isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
    device = self._execution_device

    batch_size = batch_size * num_images_per_prompt

    do_classifier_free_guidance = prior_guidance_scale > 1.0

    text_embeddings, text_encoder_hidden_states, text_mask = self._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance
    )#([2, 768]) cond 和 uncond

    # prior

    self.prior_scheduler.set_timesteps(prior_num_inference_steps, device=device)
    prior_timesteps_tensor = self.prior_scheduler.timesteps

    embedding_dim = self.prior.config.embedding_dim
    prior_latents = self.prepare_latents(
        (batch_size, embedding_dim),
        text_embeddings.dtype,
        device,
        generator,
        prior_latents,
        self.prior_scheduler,
    )

    for i, t in enumerate(self.progress_bar(prior_timesteps_tensor)):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([prior_latents] * 2) if do_classifier_free_guidance else prior_latents

        predicted_image_embedding = self.prior(
            latent_model_input,
            timestep=t,
            proj_embedding=text_embeddings,
            encoder_hidden_states=text_encoder_hidden_states,
            attention_mask=text_mask,
        ).predicted_image_embedding

        if do_classifier_free_guidance:
            predicted_image_embedding_uncond, predicted_image_embedding_text = predicted_image_embedding.chunk(2)
            predicted_image_embedding = predicted_image_embedding_uncond + prior_guidance_scale * (
                predicted_image_embedding_text - predicted_image_embedding_uncond
            )

        if i + 1 == prior_timesteps_tensor.shape[0]:
            prev_timestep = None
        else:
            prev_timestep = prior_timesteps_tensor[i + 1]

        prior_latents = self.prior_scheduler.step(
            predicted_image_embedding,
            timestep=t,
            sample=prior_latents,
            generator=generator,
            prev_timestep=prev_timestep,
        ).prev_sample

    prior_latents = self.prior.post_process_latents(prior_latents)

    image_embeddings = prior_latents
    return image_embeddings


## unclip的prior 模型 加载
prior_pipe = diffusers.UnCLIPPipeline.from_pretrained(
        "kakaobrain/karlo-v1-alpha",
        torch_dtype=torch.float16,
        # local_files_only=True,
    )
prior_pipe.decoder = None
prior_pipe.super_res_first = None
prior_pipe.super_res_last = None
prior_pipe.to(device)

prior_pipe.text_to_image_embedding = types.MethodType(karlo_prior, prior_pipe)
    

import os
if True:
    random_generator = torch.Generator(device=device).manual_seed(1000)
    """
    out_dir = "outputs_test317_0.3_blip/"
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
        
        text1 = get_txt(init_image)
        text2 = get_txt(init_image2)
        prompt = text1 + ',,,, ' + text2 #"8k resolution, best quality, remix"
        image_embeddings = prior_pipe.text_to_image_embedding(prompt, generator=random_generator)
        
        a  = unclip_image_interpolation(img_s, img_e, 3, 2023, image_embeddings=image_embeddings)
        for k in range(len(a)):
            akimg = np.concatenate([cv2_res, pil2cv(a[k])], axis=1)
            cv2.imwrite(output+'_{}.jpg'.format(k), akimg)
    """


    out_dir = 'outputs_test317_2/'
    os.makedirs(out_dir, exist_ok=True)
    testset = 'datatest/317_2'
    l1 = os.path.join(testset, 'input')
    l2 = os.path.join(testset, 'style')
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
            
            text1 = get_txt(init_image)
            text2 = get_txt(init_image2)
            prompt = text1 + ', in the style of ' + text2 #"8k resolution, best quality, remix"
            image_embeddings = prior_pipe.text_to_image_embedding(prompt, generator=random_generator)
            

            output = out_dir+'{}'.format(pair_num)
            a  = unclip_image_interpolation(img_s, img_e, 3, 2023, image_embeddings=image_embeddings)
            for k in range(len(a)):
                akimg = np.concatenate([cv2_res, pil2cv(a[k])], axis=1)
                cv2.imwrite(output+'_{}.jpg'.format(k), akimg)
            
            pair_num += 1

print("done")
