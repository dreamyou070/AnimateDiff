LORA_NAME="ufotable_lycoris_v4.safetensors"
python lora_inserting.py \
 --base_model_path ../../pretrained_stable_diffusion/stable-diffusion-v1-5 \
 --checkpoint_path pretrained/loras/DreamBooth_LoRA/${LORA_NAME} \
 --dump_path output/lora_convert_check