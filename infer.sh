# -m = run library module as a script
# config = about inference things
# inference-config = about unet structure

python -m scripts.animate --config configs/prompts/v3/v3-1-T2V.yaml \
 --pretrained-model-path ../../pretrained_stable_diffusion/stable-diffusion-v1-5 \
 --inference-config configs/inference/inference-v1.yaml