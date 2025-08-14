from functools import partial
import os
import os.path as osp
import sys
# Add the project directory to the Python path to simplify imports without manually setting PYTHONPATH.
current_dir = osp.dirname(osp.abspath(__file__)) # train_scripts
parent_dir = osp.abspath(osp.join(current_dir, "..")) # training_and_inference
sys.path.insert(0, parent_dir)
grandparent_dir = osp.abspath(osp.join(current_dir, "..", "..")) # SPO
sys.path.insert(0, grandparent_dir)

import copy
import contextlib
import math
import json

import tqdm
import random
import torch
import swanlab

from absl import app, flags
from ml_collections import config_flags
from mmengine.config import Config
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration, broadcast
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg
)
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
from peft import LoraConfig
from peft.utils import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from spo.preference_models import get_preference_model_func
from spo.datasets import build_dataset
from spo.utils import (
    huggingface_cache_dir, 
    UNET_CKPT_NAME, 
    gather_tensor_with_diff_shape,
)
from spo.custom_diffusers import (
    multi_sample_pipeline, 
    ddim_step_with_logprob,
)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", 
    "configs/spo_sd-v1-5_4k-prompts_num-sam-4_10ep_bs10.py", 
    "Training configuration."
)

logger = get_logger(__name__)


def main(_):
    config = FLAGS.config
    config = Config(config.to_dict())
    
    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=False,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="swanlab" if not getattr(config, 'debug', False) else None,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )
    if accelerator.is_main_process:
        if not getattr(config, 'debug', False):
            accelerator.init_trackers(
                project_name=config.wandb_project_name, 
                config=config, 
                init_kwargs={"swanlab": {
                    "name": config.run_name, 
                    "entity": config.wandb_entity_name
                }}
            )
        os.makedirs(os.path.join(config.logdir, config.run_name), exist_ok=True)
        with open(os.path.join(config.logdir, config.run_name, "exp_config.py"), "w") as f:
            f.write(config.pretty_text)
    logger.info(f"\n{config.pretty_text}")

    set_seed(config.seed, device_specific=True)
    
    # For mixed precision training we cast all non-trainable weigths (vae, text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # load models.
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model, 
        torch_dtype=inference_dtype,
        cache_dir=huggingface_cache_dir,
    )
    if config.use_xformers:
        pipeline.enable_xformers_memory_efficient_attention()
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    unet = pipeline.unet
    if config.use_checkpointing:
        unet.enable_gradient_checkpointing()
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=3,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Sampling Timestep",
        dynamic_ncols=True,
    )
    
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(accelerator.device)
    
    preference_model_fn = get_preference_model_func(config.preference_model_func_cfg, accelerator.device)
    if hasattr(config, 'aigi_detector_func_cfg'):
        aigi_detector_fn = get_preference_model_func(config.aigi_detector_func_cfg, accelerator.device)

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:
        unet.to(accelerator.device, dtype=inference_dtype)
        unet.requires_grad_(False)
    else:
        unet.requires_grad_(True)
    
    if config.use_lora:
        unet_lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        unet.add_adapter(unet_lora_config)
        if accelerator.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(unet, dtype=torch.float32)

    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if isinstance(models[0], type(accelerator.unwrap_model(unet))):
            if config.use_lora:
                unet_lora_layers_to_save = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(models[0])
                )
                StableDiffusionPipeline.save_lora_weights(
                    output_dir,
                    unet_lora_layers=unet_lora_layers_to_save,
                )
                logger.info(f"saved lora weights to {output_dir}")
            else:
                models[0].save_pretrained(os.path.join(output_dir, UNET_CKPT_NAME))
                logger.info(f"saved weights to {os.path.join(output_dir, UNET_CKPT_NAME)}")
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if isinstance(models[0], type(accelerator.unwrap_model(unet))):
            if config.use_lora:
                lora_state_dict, network_alphas = StableDiffusionPipeline.lora_state_dict(input_dir)
                unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
                unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
                incompatible_keys = set_peft_model_state_dict(models[0], unet_state_dict, adapter_name="default")
                if incompatible_keys is not None:
                    # check only for unexpected keys
                    unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                    if unexpected_keys:
                        logger.warning(
                            f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                            f" {unexpected_keys}. "
                        )
                if accelerator.mixed_precision == "fp16":
                    # only upcast trainable parameters (LoRA) into fp32
                    cast_training_params([models[0]], dtype=torch.float32)
                logger.info(f"loaded lora weights from {input_dir}")                
            else:
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder=UNET_CKPT_NAME)
                models[0].register_to_config(**load_model.config)
                models[0].load_state_dict(load_model.state_dict())
                logger.info(f"loaded weights from {input_dir}")                
                del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    
    trainable_para = filter(lambda p: p.requires_grad, unet.parameters())
    optimizer = optimizer_cls(
        trainable_para,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    prompt_dataset = build_dataset(config.dataset_cfg)
    collate_fn = partial(
        prompt_dataset.collate_fn,
        tokenizer=pipeline.tokenizer,
    )

    data_loader = torch.utils.data.DataLoader(
        prompt_dataset,
        collate_fn=collate_fn,
        batch_size=config.sample.sample_batch_size,
        num_workers=config.dataloader_num_workers,
        shuffle=config.dataloader_shuffle,
        pin_memory=config.dataloader_pin_memory,
        drop_last=config.dataloader_drop_last,
    )
    
    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    # for some reason, autocast is necessary for non-lora training but not for lora training, and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    
    # Prepare everything with `accelerator`.
    unet, optimizer, data_loader = accelerator.prepare(unet, optimizer, data_loader)
        
    # Train!
    total_train_batch_size = (
        config.train.train_batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps
    )
    num_update_steps_per_epoch = math.ceil(len(data_loader) / config.train.gradient_accumulation_steps)
    config.num_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)
    
    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Max Train Steps = {config.max_train_steps}")
    logger.info(f"  Sampling batch size per device = {config.sample.sample_batch_size}")
    logger.info(f"  Training batch size per device = {config.train.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Do Classifier Free Guidance = {config.train.cfg}")
    if hasattr(config, 'aigi_detector_func_cfg'):
        logger.info(f"  Aigi Detector Weight: {config.aigi_detector_weight}")
        logger.info(f"  Aigi Detector: {config.aigi_detector_func_cfg.aigi_detector}")
        logger.info(f"  Aigi Detector Path: {config.aigi_detector_func_cfg.aigi_detector_path}")
    logger.info("")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
        with open(os.path.join(config.resume_from, "global_step.json"), "r") as f:
            global_step = json.load(f)["global_step"]
    else:
        first_epoch = 0
        global_step = 0
    
    train_step_progress_bar = tqdm(
            range(0, config.max_train_steps),
            total=config.max_train_steps,
            initial=global_step,
            disable=not accelerator.is_local_main_process,
            desc="Train steps",
            position=0)
    
    for epoch in tqdm(
        range(first_epoch, config.num_epochs),
        total=config.num_epochs,
        initial=first_epoch,
        disable=not accelerator.is_local_main_process,
        desc="Epoch",
        position=1,
    ):
        train_scores, train_backward_loss = 0.0, 0.0
        if hasattr(config, "aigi_detector_func_cfg"):
            train_aigi_detector_scores = 0.0
        
        data_loader_process_bar = tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            disable=not accelerator.is_local_main_process,
            desc="Batch",
            position=2,
        )
        for dataset_batch_idx, batch in data_loader_process_bar:
            ############ Training ############
            pipeline.unet.train()
            
            # ############ pre-define the parameters of drtune. ############
            K = config.drtune.K
            T = config.drtune.T
            M = config.drtune.M
            interval = T // K
            s = random.randint(0, T - K * interval - 1) if T % K != 0 else random.randint(0, T - (K-1) * interval - 1)
            train_timestep_indices = [ s + i * interval for i in range(0, K)]
            t_min = random.randint(1, M) # range of early stop timestep M.
            
            # ############ pre-define the parameters of stable_diffusion_pipeline.__call__ ############
            prompt, negative_prompt, generator, latents, cross_attention_kwargs = None, None, None, None, None
            callback_steps, num_images_per_prompt = 1, 1
            guidance_rescale = 0.0
            eta = config.train.eta
            guidance_scale = config.train.guidance_scale
            num_inference_steps = config.drtune.T
            
            do_classifier_free_guidance = config.train.cfg
            
            batch_size = batch['input_ids'].shape[0]
            prompt_ids = batch['input_ids']
            # encode prompts
            prompt_embeds = pipeline.text_encoder(prompt_ids)[0]
            negative_prompt_embeds = neg_prompt_embed.repeat(batch_size, 1, 1)
            
            # 0. Default height and width to unet
            height = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
            width = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
            
            # 1. Check inputs. Raise error if not correct
            pipeline.check_inputs(prompt=prompt, height=height, width=width, callback_steps=callback_steps, negative_prompt=negative_prompt, 
                                    prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds)
            
            # 2. Define call parameters
            batch_size = prompt_embeds.shape[0]
            device = pipeline._execution_device
            
            # 3. Encode input prompt
            text_encoder_lora_scale = cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
            prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
            )
            
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
                
            # 4. Prepare timesteps
            pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = pipeline.scheduler.timesteps
            
            # 5. Prepare latent variables
            num_channels_latents = pipeline.unet.config.in_channels
            latents = pipeline.prepare_latents(
                batch_size=batch_size * num_images_per_prompt,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                dtype=prompt_embeds.dtype,
                device=device,
                generator=generator,
                latents=latents,
            )
            
            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)
            
            # 7. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * pipeline.scheduler.order

            with accelerator.accumulate(pipeline.unet):
                with autocast():
                    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
                        for i, t in enumerate(timesteps):
                            # expand the latents if we are doing classifier free guidance
                            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

                            # predict the noise residual
                            noise_pred = pipeline.unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=prompt_embeds,
                                cross_attention_kwargs=cross_attention_kwargs,
                                return_dict=False,
                            )[0]
                            
                            # perform guidance
                            if do_classifier_free_guidance:
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                            if do_classifier_free_guidance and guidance_rescale > 0.0:
                                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
                            
                            if i not in train_timestep_indices:
                                noise_pred = noise_pred.detach()
                                
                            # compute the previous noisy sample x_t -> x_t-1
                            step_output = pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)
                            latents = step_output.prev_sample
                            pred_original_sample = step_output.pred_original_sample
                            
                            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                                progress_bar.update()
                                
                            if num_inference_steps - i == t_min:
                                break
                    
                # translate latent to image.
                pred_original_sample = pred_original_sample / pipeline.vae.config.scaling_factor  
                pred_x0 = pipeline.vae.decode(pred_original_sample.to(pipeline.vae.dtype)).sample
                
                scores = preference_model_fn(pred_x0, batch['extra_info']).mean()
                backward_loss = -1 * scores
                
                if hasattr(config, "aigi_detector_func_cfg"):
                    aigi_detector_scores = aigi_detector_fn(pred_x0, batch['extra_info']).mean()
                    backward_loss = (1 - config.aigi_detector_weight) * (-1 * scores) +  config.aigi_detector_weight * (-1 * aigi_detector_scores)

                avg_scores = accelerator.reduce(scores.detach(), reduction="mean")
                avg_backward_loss = accelerator.reduce(backward_loss.detach(), reduction="mean")
                
                train_scores += avg_scores / accelerator.gradient_accumulation_steps
                train_backward_loss += avg_backward_loss / accelerator.gradient_accumulation_steps
                
                if hasattr(config, "aigi_detector_func_cfg"):
                    avg_aigi_detector_scores = accelerator.reduce(aigi_detector_scores.detach(), reduction="mean")
                    train_aigi_detector_scores += avg_aigi_detector_scores / accelerator.gradient_accumulation_steps

                # backward pass
                accelerator.backward(backward_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_para, config.train.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                
            if accelerator.sync_gradients:
                # log training-related stuff
                info = {
                    "epoch": epoch, 
                    "global_step": global_step, 
                    "train_scores": train_scores,
                    "train_backward_loss": train_backward_loss,
                    "lr": optimizer.param_groups[0]['lr'],
                }
                
                if hasattr(config, "aigi_detector_func_cfg"):
                    info["train_aigi_detector_socres"] = train_aigi_detector_scores
                    train_aigi_detector_scores = 0.0
                    
                accelerator.log(info, step=global_step)
                global_step += 1
                train_step_progress_bar.update(1)
                train_scores, train_backward_loss = 0.0, 0.0

            
        ########## save ckpt and evaluation ##########
        if accelerator.is_main_process:
            if (epoch + 1) % config.save_interval == 0:
                accelerator.save_state(os.path.join(config.logdir, config.run_name, f'checkpoint_{epoch}'))
                with open(os.path.join(config.logdir, config.run_name, f'checkpoint_{global_step}', 'global_step.json'), 'w') as f:
                    json.dump({'global_step': global_step}, f)
                    
            if (epoch + 1) % config.eval_interval == 0 and config.validation_prompts is not None:
                prompt_info = f"Running validation... \n Generating {config.num_validation_images} images with prompt:\n"
                for prompt in config.validation_prompts:
                    prompt_info = prompt_info + prompt + '\n'

                logger.info(prompt_info)
                # create pipeline
                pipeline.unet.eval()
                # run inference
                generator = torch.Generator(device=accelerator.device).manual_seed(config.seed) if config.seed else None

                image_logs = []
                for idx, validation_prompt in enumerate(config.validation_prompts):
                    with torch.cuda.amp.autocast():
                        images = [
                            pipeline(
                                prompt=validation_prompt,
                                num_inference_steps=config.pipeline_num_inference_steps,
                                generator=generator,
                                guidance_scale=config.sample.guidance_scale,
                            ).images[0]
                            for _ in range(config.num_validation_images)
                        ]
                    image_logs.append(
                        {
                            "images": images, 
                            "prompts": validation_prompt,
                        }
                    )

                for tracker in accelerator.trackers:
                    if tracker.name == "swanlab":
                        formatted_images = []
                        for log in image_logs:
                            images = log["images"]
                            validation_prompt = log["prompts"]
                            for idx, image in enumerate(images):
                                image = swanlab.Image(image, caption=validation_prompt)
                                formatted_images.append(image)
                        tracker.log({"validation": formatted_images}, step=global_step)
                
                pipeline.unet.train()
                torch.cuda.empty_cache()
    
    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=os.path.join(config.logdir, config.run_name),
            unet_lora_layers=unet_lora_state_dict,
        )
    
    accelerator.end_training()

if __name__ == "__main__":
    app.run(main)
