# -m = run library module as a script
# config = about inference things
# inference-config = about unet structure
#--FPS ${Frames_per_second} \

number_of_frame=16
Frames_per_second=8

python scripts/animate.py \
 --pretrained-model-path "../../pretrained/stable-diffusion-v1-5" \
 --config configs/prompts/v4/v4-1-mistoon.yaml \
 --inference-config configs/inference/inference-v1.yaml \
 --H 256 \
 --W 256 \
 --L ${number_of_frame} \
 --without-xformers