python test_latency.py \
--model-path /path/to/models/Qwen3-8B \
--tp 8 \
--input-len 8192 \
--output-len 8192 \
--batch-size 64 \
--num-iters-warmup 0 \
--tbik \
--num-iters 3 \
--output-json /path/to/save/results.json