set -e
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
accelerate launch ./train_stage_1.py
# while true; do
#     accelerate launch ./train_stage_1.py || true
# done