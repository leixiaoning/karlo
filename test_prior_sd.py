import types
from typing import Union, List, Optional, Callable

import diffusers
import torch

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput


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
    )

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


@torch.inference_mode()
def sd_image_variations_decoder(
    self,
    image_embeddings,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: Optional[int] = 1,
):
    """
    copy from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_image_variation.py#L289
    """
    # 0. Default height and width to unet
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    batch_size = len(image_embeddings)

    device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    image_embeddings = image_embeddings.unsqueeze(1)

    # duplicate image embeddings for each generation per prompt, using mps friendly method
    bs_embed, seq_len, _ = image_embeddings.shape
    image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
    image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

    if do_classifier_free_guidance:
        uncond_embeddings = torch.zeros_like(image_embeddings)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        image_embeddings = torch.cat([uncond_embeddings, image_embeddings])

    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = self.unet.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        image_embeddings.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=image_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    # 8. Post-processing
    image = self.decode_latents(latents)

    # 9. Run safety checker
    image, has_nsfw_concept = self.run_safety_checker(image, device, image_embeddings.dtype)

    # 10. Convert to PIL
    if output_type == "pil":
        image = self.numpy_to_pil(image)

    if not return_dict:
        return (image, has_nsfw_concept)

    return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


def main():
    device = "cuda:0"

    free_gm, total_gm = torch.cuda.mem_get_info()
    print(f"begin: GPU MEM: {(total_gm - free_gm) / (2 ** 30):.2f}G/{total_gm / (2 ** 30):.2f}G")

    decoder_pipe = diffusers.StableDiffusionImageVariationPipeline.from_pretrained(
        "/www/simple_ssd/lxn3/data_download/sd-image-variations-diffusers",#"lambdalabs/sd-image-variations-diffusers",
        revision="v2.0",
        torch_dtype=torch.float16,
        # local_files_only=True,
        safety_checker=None,
    )
    decoder_pipe.image_encoder = None #CLIPVisionModelWithProjection 
    decoder_pipe.to(device)

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
    decoder_pipe.image_embedding_to_image = types.MethodType(sd_image_variations_decoder, decoder_pipe)

    free_gm, total_gm = torch.cuda.mem_get_info()
    print(f"after load models: GPU MEM: {(total_gm - free_gm) / (2 ** 30):.2f}G/{total_gm / (2 ** 30):.2f}G")

    random_generator = torch.Generator(device=device).manual_seed(1000)

    prompt = "a shiba inu wearing a beret and black turtleneck"
    image_embeddings = prior_pipe.text_to_image_embedding(prompt, generator=random_generator)

    image = decoder_pipe.image_embedding_to_image(image_embeddings, generator=random_generator).images[0]

    image.save("./shiba-inu.jpg")


if __name__ == "__main__":
    main()