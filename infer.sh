# -m = run library module as a script
# config = about inference things
# inference-config = about unet structure

python scripts/animate.py \
 --pretrained-model-path ../../pretrained_stable_diffusion/stable-diffusion-v1-5 \
 --config configs/prompts/v3/v3-1-T2V.yaml \
 --inference-config configs/inference/inference-v1.yaml \
 --H 512 \
 --W 512 \
 --L 16 \
 --without-xformers