import argparse
import inspect
import datetime
import os
from omegaconf import OmegaConf
import torch
import torchvision.transforms as transforms
import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
# how to find animatediff ?
# add present dir on sys
import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(parent_dir)
from animatediff.models.unet import UNet3DConditionModel
from animatediff.models.sparse_controlnet import SparseControlNetModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.util import load_weights
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange, repeat
import csv, pdb, glob, math
from pathlib import Path
from PIL import Image
import numpy as np


@torch.no_grad()
def main(args):

    print(f"\n step 1. argument check")
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)

    print(f"\n step 2. start of inference")
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = os.path.join('samples', f'{Path(args.config).stem}-{time_str}')
    os.makedirs(savedir, exist_ok=True)

    print(f"\n step 3. check config") # OmegaConf.load (read config yaml)
    config = OmegaConf.load(args.config) # load # why it cannot read configuratio of model?

    # what is config ???
    samples = []

    print(f"\n step 4. loading model")
    device = args.device
    # loading model
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device)

    print(f"\n step 5. inference")
    sample_idx = 0
    for model_idx, model_config in enumerate(config):
        # model config = v3-1-T2V.yaml
        print(f' (5.{model_idx}) model inference')
        # if there is no config, use argument
        # model_config.get "W" =
        print(f'model_config = {model_config}')
        model_config.W = model_config.get("W", args.W)
        model_config.H = model_config.get("H", args.H)
        model_config.L = model_config.get("L", args.L) # 16

        # ------------------------------------------------------------------------------------------------------------------------
        # [2] inference config
        # anomal yaml file (configs/inference/inference-v3.yaml)
        # to read yaml file, OmegaConf is necessary
        inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))

        # ------------------------------------------------------------------------------------------------------------------------
        # [3] make unet model
        # making 3D unet model
        unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path,
                                                       subfolder="unet",
                                                       unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)).to(device)

        # ------------------------------------------------------------------------------------------------------------------------
        # [4] controlnet
        # what kind of controlnet to use ... ?
        # what is controlnet image ... ?

        controlnet = controlnet_images = None # scribble
        if model_config.get("controlnet_path", "") != "":
            print(f' controlnet start !')
            assert model_config.get("controlnet_images", "") != ""
            assert model_config.get("controlnet_config", "") != ""
            # make new attribute
            unet.config.num_attention_heads = 8
            unet.config.projection_class_embeddings_input_dim = None
            controlnet_config = OmegaConf.load(model_config.controlnet_config)
            controlnet = SparseControlNetModel.from_unet(unet,
                                                         controlnet_additional_kwargs=controlnet_config.get("controlnet_additional_kwargs", {}))
            print(f"loading controlnet checkpoint from {model_config.controlnet_path} ...")
            controlnet_state_dict = torch.load(model_config.controlnet_path, map_location="cpu")
            controlnet_state_dict = controlnet_state_dict["controlnet"] if "controlnet" in controlnet_state_dict else controlnet_state_dict
            controlnet_state_dict.pop("animatediff_config", "")
            controlnet.load_state_dict(controlnet_state_dict)
            controlnet.to(device)

            # ----------------------------------------------------------------------------------------------------------
            # controlnet_images ?
            image_paths = model_config.controlnet_images # more than two ??
            if isinstance(image_paths, str) :
                image_paths = [image_paths]
            # why there is no controlnet_images ?
            assert len(image_paths) <= model_config.L

            # ----------------------------------------------------------------------------------------------------------
            # here, L = if there is no L ..
            image_transforms = transforms.Compose([transforms.RandomResizedCrop((model_config.H, model_config.W), (1.0, 1.0),
                                                                                ratio=(model_config.W/model_config.H, model_config.W/model_config.H)),
                                                   transforms.ToTensor(),])
            if model_config.get("normalize_condition_images", False):
                def image_norm(image):
                    image = image.mean(dim=0, keepdim=True).repeat(3,1,1)
                    image -= image.min()
                    image /= image.max()
                    return image
            else:
                image_norm = lambda x: x # no normalizing mappint
            controlnet_images = [image_norm(image_transforms(Image.open(path).convert("RGB"))) for path in image_paths]
            os.makedirs(os.path.join(savedir, "control_images"), exist_ok=True)
            for i, image in enumerate(controlnet_images):
                Image.fromarray((255. * (image.numpy().transpose(1,2,0))).astype(np.uint8)).save(f"{savedir}/control_images/{i}.png")
            controlnet_images = torch.stack(controlnet_images).unsqueeze(0).to(device)
            controlnet_images = rearrange(controlnet_images, "b f c h w -> b c f h w")
            # ----------------------------------------------------------------------------------------------------------
            # original range = batch, frame = 1, channel = 3, h, w
            # rearranged = batch, channel, frame = 2, H, W
            # [3] arranging controlnet images
            if controlnet.use_simplified_condition_embedding:
                num_controlnet_images = controlnet_images.shape[2]
                controlnet_images = rearrange(controlnet_images, "b c f h w -> (b f) c h w")
                controlnet_images = vae.encode(controlnet_images * 2. - 1.).latent_dist.sample() * 0.18215
                controlnet_images = rearrange(controlnet_images, "(b f) c h w -> b c f h w", f=num_controlnet_images)
                # vae prepared control image
        # set xformers
        if is_xformers_available() and (not args.without_xformers):
            unet.enable_xformers_memory_efficient_attention()
            if controlnet is not None: controlnet.enable_xformers_memory_efficient_attention()

        # [3] pure pipeline
        pipeline = AnimationPipeline(vae=vae,
                                     text_encoder=text_encoder,
                                     tokenizer=tokenizer,
                                     unet=unet,
                                     controlnet=controlnet,
                                     scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),) #.to("cuda")

        # [4] pipeline with motion module / lora / domain adapter
        pipeline = load_weights(pipeline,
                                motion_module_path         = model_config.get("motion_module", ""),                 # motion module
                                motion_module_lora_configs = model_config.get("motion_module_lora_configs", []),
                                adapter_lora_path          = model_config.get("adapter_lora_path", ""),             # domain adapter
                                adapter_lora_scale         = model_config.get("adapter_lora_scale", 1.0),
                                dreambooth_model_path      = model_config.get("dreambooth_path", ""),
                                lora_model_path            = model_config.get("lora_model_path", ""),
                                lora_alpha                 = model_config.get("lora_alpha", 0.8),).to(device)

        # -----------------------------------------------------------------------------------------------------------------------
        # [5] inference
        prompts = model_config.prompt
        n_prompts = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt
        random_seeds = model_config.get("seed", [-1])
        random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
        random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
        config[model_idx].random_seed = []


        for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):


            # make two video .. ?
            # one by one
            # manually set random seed for reproduction
            if random_seed != -1:
                torch.manual_seed(random_seed)
            else:
                torch.seed()
            print(f'model_idx = {model_idx}')
            config[model_idx].random_seed.append(torch.initial_seed())



            # ------------------------------------------------------------------------------------------------------------
            # There are 16 frames (pipeline is AnimationPipeline) !
            sample = pipeline(prompt,
                              negative_prompt     = n_prompt,
                              num_inference_steps = model_config.steps,
                              guidance_scale      = model_config.guidance_scale,
                              width               = model_config.W, # 512
                              height              = model_config.H, # 256
                              video_length        = model_config.L,
                              controlnet_images = controlnet_images, # controlnet here
                              controlnet_image_index = model_config.get("controlnet_image_indexs", [0]),).videos
            samples.append(sample)

            prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
            save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{prompt}.gif")
            print(f"save to {savedir}/sample/{prompt}.gif")
            
            sample_idx += 1

    samples = torch.concat(samples)
    save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)
    OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, default="pretrained/models/StableDiffusion/stable-diffusion-v1-5",)
    parser.add_argument("--inference-config",      type=str, default="configs/inference/inference-v1.yaml")    
    parser.add_argument("--config",                type=str, required=True)
    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--without-xformers", action="store_true")
    parser.add_argument('--device', type = str, default= 'cuda')
    args = parser.parse_args()
    main(args)