# coding=utf-8
# Copyright 2023, Haofan Wang, Qixun Wang, All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
#  Changes were made to this source code by Yuwei Guo.
""" Conversion script for the LoRA's safetensors checkpoints. """

import argparse

import torch
from safetensors.torch import load_file

from diffusers import StableDiffusionPipeline



def main(args) :

    print(f'\n step 1. make model')
    base_model_path = args.base_model_path
    checkpoint_path = args.checkpoint_path
    lora_prefix_unet = args.lora_prefix_unet
    lora_prefix_text_encoder = args.lora_prefix_text_encoder
    alpha = args.alpha

    print(f'\n step 2. make pipeline as convert model name')
    from animatediff.utils.convert_lora_safetensor_to_diffusers import convert
    try :
        pipeline = StableDiffusionPipeline.from_pretrained(base_model_path,
                                                           torch_dtype=torch.float32)
        pipeline.load_lora_weights("lordjia/by-feng-zikai",
                                   weight_name=checkpoint_path,
                                   adapter_name="feng")
        pipe = convert(base_model_path,
                       checkpoint_path,
                       lora_prefix_unet,
                            lora_prefix_text_encoder,
                            alpha)
        pipe = pipe.to(args.device)
    except :
        pipeline = StableDiffusionPipeline.from_pretrained(base_model_path,
                                                           torch_dtype=torch.float32)
        # change weight dtype


    pipe.save_pretrained(args.dump_path, safe_serialization=args.to_safetensors)

# lora inserting
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", default=None, type=str, required=True, help="Path to the base model in diffusers format.")
    parser.add_argument("--checkpoint_path", default=None, type=str, required=True, help="Path to the checkpoint to convert.")
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")
    parser.add_argument("--lora_prefix_unet", default="lora_unet", type=str, help="The prefix of UNet weight in safetensors")
    parser.add_argument("--lora_prefix_text_encoder",default="lora_te",type=str,
                        help="The prefix of text encoder weight in safetensors",)
    parser.add_argument("--alpha", default=0.75, type=float, help="The merging ratio in W = W0 + alpha * deltaW")
    parser.add_argument("--to_safetensors", action="store_true", help="Whether to store pipeline in safetensors format or not.")
    parser.add_argument("--device", type=str, help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")

    args = parser.parse_args()
    main(args)
