export VLLM_ATTENTION_BACKEND="TRITON_ATTN"
export VLLM_BATCH_INVARIANT='1'
export VLLM_TP_INVARIANT='0'
export ALIGN_TRAIN_INFERENCE='1'

python -m rl_torchtitan_vllm.driver \
  --model-name "Qwen/Qwen3-1.7B" \
  --use-real-dataset \
  --max-model-len 40960 \
  --vllm-port 12346 \
  --run-dir "outputs/rl_fsdp_tbik_beta0.5" \
  --max-new-tokens 512 \
  --num-train-samples 256 \
  --num-test-samples 64 \
  --num-steps 100 \
  --group-size 8 \
  --rollout-batch-size 4 \
  --train-micro-batch-size 8 \
  --eval-every-n-steps 10 \
  --num-eval-per-sample 4 \
  --vllm-gpu-memory-utilization 0.6 \
  --rollout-gpus 0,1,2,3 \
  --train-gpus 0,1,2,3 \
  --grpo-beta 0.5 \
  --use-vllm-compat \
  --resume
  
