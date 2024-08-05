set -e
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
export 'CUDA_VISIBLE_DEVICES=0'
accelerate launch ./train_variant_3.py
# while true; do
#     accelerate launch ./train_variant_3.py || true
# done