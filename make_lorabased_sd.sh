LORA_NAME="mistoonAnime_v30.safetensors"
python make_lorabased_sd.sh.py \
 --base_model_path ../../pretrained_stable_diffusion/stable-diffusion-v1-5 \
 --checkpoint_path pretrained/loras/DreamBooth_LoRA/${LORA_NAME} \
 --dump_path pretrained/models/StableDiffusion/BambaBaby