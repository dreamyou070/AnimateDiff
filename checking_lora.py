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


def load_diffusers_lora(pipeline, state_dict, alpha=1.0):
    # directly update weight in diffusers model
    for key in state_dict:
        # only process lora down key
        if "up." in key: continue

        up_key = key.replace(".down.", ".up.")
        model_key = key.replace("processor.", "").replace("_lora", "").replace("down.", "").replace("up.", "")
        model_key = model_key.replace("to_out.", "to_out.0.")
        layer_infos = model_key.split(".")[:-1]

        curr_layer = pipeline.unet
        while len(layer_infos) > 0:
            temp_name = layer_infos.pop(0)
            curr_layer = curr_layer.__getattr__(temp_name)

        weight_down = state_dict[key]
        weight_up = state_dict[up_key]
        curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).to(curr_layer.weight.data.device)

    return pipeline


def convert_lora(pipeline, state_dict, LORA_PREFIX_UNET="lora_unet", LORA_PREFIX_TEXT_ENCODER="lora_te", alpha=0.6):
    # load base model

    # pipeline = StableDiffusionPipeline.from_pretrained(base_model_path, torch_dtype=torch.float32)

    # load LoRA weight from .safetensors
    # state_dict = load_file(checkpoint_path)

    visited = []

    # [1] check civitat state_dict
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"
        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3).to(
                curr_layer.weight.data.device)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).to(curr_layer.weight.data.device)

        # update visited list
        for item in pair_keys:
            visited.append(item)

    return pipeline


def convert(base_model_path, checkpoint_path, LORA_PREFIX_UNET, LORA_PREFIX_TEXT_ENCODER, alpha):
    # load base model
    pipeline = StableDiffusionPipeline.from_pretrained(base_model_path, torch_dtype=torch.float32)

    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path)

    visited = []
    # directly update weight in diffusers model
    for key in state_dict:
        # key = cond_stage_model.transformer.text_model.encoder.layers.1.mlp.fc2.weight
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            # there is no lora_te ...

            layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)

        # ------------------------------------------------------------------------------------------------------------------------
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)
        # ------------------------------------------------------------------------------------------------------------------------
        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        """
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            print(f'weight_up   = {weight_up.shape}')
            print(f'weight_down = {weight_down.shape}')
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)


        #
        if len(state_dict[pair_keys[0]].shape.size()) == 2:
            # linear
            if len(up_weight.size()) == 4:  # use linear projection mismatch
                up_weight = up_weight.squeeze(3).squeeze(2)
                down_weight = down_weight.squeeze(3).squeeze(2)
            curr_layer.weight.data += alpha * (up_weight @ down_weight)
        elif down_weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            curr_layer.weight.data += alpha * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            # conv2d 3x3
            conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
            # logger.info(conved.size(), weight.size(), module.stride, module.padding)
            curr_layer.weight.data += alpha * conved

        # update visited list
        for item in pair_keys:
            visited.append(item)
        """

    return pipeline

checkpoint_path = r'pretrained/loras/DreamBooth_LoRA/ufotable_lycoris_v4.safetensors'
state_dict = load_file(checkpoint_path)
visited = []
for key in state_dict:
    # key = cond_stage_model.transformer.text_model.encoder.layers.1.mlp.fc2.weight
    # it is suggested to print out the key, it usually will be something like below
    # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

    # as we have set the alpha beforehand, so just skip
    if ".alpha" in key or key in visited:
        continue

    if "text" in key:
        # there is no lora_te ...

        layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
        curr_layer = pipeline.text_encoder
    else:
        layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
        curr_layer = pipeline.unet

    # find the target layer
    temp_name = layer_infos.pop(0)

    # ------------------------------------------------------------------------------------------------------------------------
    while len(layer_infos) > -1:
        try:
            curr_layer = curr_layer.__getattr__(temp_name)
            if len(layer_infos) > 0:
                temp_name = layer_infos.pop(0)
            elif len(layer_infos) == 0:
                break
        except Exception:
            if len(temp_name) > 0:
                temp_name += "_" + layer_infos.pop(0)
            else:
                temp_name = layer_infos.pop(0)
    # ------------------------------------------------------------------------------------------------------------------------
    pair_keys = []
    if "lora_down" in key:
        pair_keys.append(key.replace("lora_down", "lora_up"))
        pair_keys.append(key)
    else:
        pair_keys.append(key)
        pair_keys.append(key.replace("lora_up", "lora_down"))

    # update weight
    """
    if len(state_dict[pair_keys[0]].shape) == 4:
        weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
        weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
        print(f'weight_up   = {weight_up.shape}')
        print(f'weight_down = {weight_down.shape}')
        curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
    else:
        weight_up = state_dict[pair_keys[0]].to(torch.float32)
        weight_down = state_dict[pair_keys[1]].to(torch.float32)
        curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)


    #
    if len(state_dict[pair_keys[0]].shape.size()) == 2:
        # linear
        if len(up_weight.size()) == 4:  # use linear projection mismatch
            up_weight = up_weight.squeeze(3).squeeze(2)
            down_weight = down_weight.squeeze(3).squeeze(2)
        curr_layer.weight.data += alpha * (up_weight @ down_weight)
    elif down_weight.size()[2:4] == (1, 1):
        # conv2d 1x1
        curr_layer.weight.data += alpha * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
    else:
        # conv2d 3x3
        conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
        # logger.info(conved.size(), weight.size(), module.stride, module.padding)
        curr_layer.weight.data += alpha * conved

    # update visited list
    for item in pair_keys:
        visited.append(item)
    """
