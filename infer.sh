# -m = run library module as a script
# config = about inference things
# inference-config = about unet structure

# ufotable style

python scripts/animate.py \
 --pretrained-model-path "/share0/dreamyou070/dreamyou070/AnimateDiff/AnimateDiff/pretrained/models/StableDiffusion/BambaBaby" \
 --config configs/prompts/v3/v3-BambaBaby.yaml \
 --inference-config configs/inference/inference-v1.yaml \
 --H 256 \
 --W 512 \
 --L 16 \
 --without-xformers