# -m = run library module as a script
# config = about inference things
# inference-config = about unet structure

# ufotable style

python scripts/animate.py \
 --pretrained-model-path ../../output/lora_unfortable_style \
 --config configs/prompts/v3/v3-1-T2V.yaml \
 --inference-config configs/inference/inference-v1.yaml \
 --H 512 \
 --W 512 \
 --L 16 \
 --without-xformers