import sys
import os
import argparse
import time
import json
import numpy as np
from tqdm import tqdm
import torch

import multiprocessing as mp

import sglang as sgl

def add_cli_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--num-iters-warmup", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=30)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--bio", action="store_true", help="Use BIO")
    parser.add_argument("--tbik", action="store_true", help="Use TBik")

def main(args: argparse.Namespace):
    np.random.seed(42)
    
    print(f"Initializing SGLang Engine with model: {args.model_path}")
    
    if args.bio:
        print('Enable Batch Invariant Mode')
    if args.tbik:
        print('Enable TP Invariant Mode')
    
    engine = sgl.Engine(
        model_path=args.model_path,
        dtype=torch.bfloat16, 
        tp_size=args.tp,  # 支持张量并行
        trust_remote_code=True,
        # disable_cuda_graph=True,
        enable_deterministic_inference=args.bio,
        rl_on_policy_target="fsdp_tp" if args.tbik else None,
        # enable
        device="cuda"
    )

    dummy_prompt_token_ids = np.random.randint(100, 30000, size=(args.batch_size, args.input_len)).tolist()

    # 构造采样参数
    sampling_params = {
        "max_new_tokens": args.output_len,
        "temperature": 0,
        "ignore_eos": True 
    }

    def run_to_completion():
        start_time = time.perf_counter()
    
        outputs = engine.generate(
            input_ids=dummy_prompt_token_ids,
            sampling_params=sampling_params
        )
        
        end_time = time.perf_counter()
        return end_time - start_time

    # Warmup
    print(f"Warming up ({args.num_iters_warmup} iters)...")
    for _ in tqdm(range(args.num_iters_warmup)):
        run_to_completion()

    # Benchmark
    print(f"Benchmarking ({args.num_iters} iters)...")
    latencies = []
    for _ in tqdm(range(args.num_iters)):
        latencies.append(run_to_completion())

    latencies = np.array(latencies)
    percentages = [10, 25, 50, 75, 90, 99]
    percentiles = np.percentile(latencies, percentages)

    print(f"\nResults (Batch Size: {args.batch_size}, In: {args.input_len}, Out: {args.output_len})")
    print(f"Avg latency: {np.mean(latencies):.4f} s")
    print(f"Throughput: {args.batch_size * args.output_len / np.mean(latencies):.2f} tokens/s")
    print("-" * 30)
    for p, val in zip(percentages, percentiles):
        print(f"{p}% percentile latency: {val:.4f} s")

    if args.output_json:
        results = {
            "avg_latency": float(np.mean(latencies)),
            "latencies": latencies.tolist(),
            "percentiles": dict(zip(map(str, percentages), percentiles.tolist())),
            "throughput_tokens_per_s": float(args.batch_size * args.output_len / np.mean(latencies))
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
            
    engine.shutdown()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args()
    main(args)