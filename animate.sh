# -m = run library module as a script
# config = about inference things
# inference-config = about unet structure
#--FPS ${Frames_per_second} \

number_of_frame=16
Frames_per_second=8
pretrained_model_path="../../pretrained_stable_diffusion/stable-diffusion-v1-5"

python animate.py \
 --pretrained_model_path "${pretrained_model_path}" \
 --config configs/prompts/v4/v4-1-mistoon.yaml \
 --inference-config configs/inference/inference-v1.yaml \
 --H 256 \
 --W 256 \
 --L ${number_of_frame} \
 --without-xformers