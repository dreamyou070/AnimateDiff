# -m = run library module as a script
# config = about inference things
# inference-config = about unet structure

number_of_frame=16
Frames_per_second=8

python scripts/animate.py \
 --pretrained-model-path "../../pretrained/models/StableDiffusion/BambaBaby" \
 --config configs/prompts/v3/v3-BambaBaby.yaml \
 --inference-config configs/inference/inference-v1.yaml \
 --H 128 \
 --W 256 \
 --L ${number_of_frame} \
 --FPS ${Frames_per_second} \
 --without-xformers