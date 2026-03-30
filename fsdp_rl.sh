export VLLM_ATTENTION_BACKEND="TRITON_ATTN"
export VLLM_BATCH_INVARIANT='1'
export VLLM_TP_INVARIANT='1'
export ALIGN_TRAIN_INFERENCE='1'

python -m rl_torchtitan_vllm.driver \
  --use-real-dataset \
  --num-steps 100 \
  --group-size 8 \
  --max-model-len 40960 \
  --max-new-tokens 512 \
  --num-dataset-samples 16 \
  --num-rollout-batches 1 \
  --train-micro-batch-size 4 \
  --vllm-gpu-memory-utilization 0.8 \
  --rollout-gpus 0,1,2,3 \
  --train-gpus 0,1,2,3 \
  --use-vllm-compat
