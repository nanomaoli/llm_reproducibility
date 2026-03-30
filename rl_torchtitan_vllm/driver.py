from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import tempfile
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from rl_torchtitan_vllm.common import (
    atomic_write_json,
    build_default_prompts,
    collect_patch_env,
    detect_vllm_compat_mode,
    download_and_convert_model,
    ensure_dir,
    format_patch_env,
    load_gsm8k_dataset,
    load_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-command driver for vLLM rollout + FSDP training")
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--model-path")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--cache-dir", default="./models")
    parser.add_argument("--output-dir", default="./converted")
    parser.add_argument("--run-dir", default="./outputs/rl_fsdp_vllm")
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--num-rollout-batches", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--train-micro-batch-size", type=int, default=0)
    parser.add_argument("--grpo-beta", type=float, default=0.1)
    parser.add_argument("--use-stable-grpo", action="store_true")
    parser.add_argument("--use-real-dataset", action="store_true")
    parser.add_argument("--use-vllm-compat", action="store_true")
    parser.add_argument("--num-dataset-samples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rollout-gpus", default="0,1,2,3")
    parser.add_argument("--train-gpus", default="0,1,2,3")
    parser.add_argument("--vllm-port", type=int)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--logdir", default="./outputs/rl_training")
    return parser.parse_args()


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def request_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 600.0) -> dict[str, Any]:
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, method=method, data=body, headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {error_body}") from exc


def wait_for_server(base_url: str, timeout_s: float = 180.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            payload = request_json("GET", f"{base_url}/health", timeout=5.0)
            if payload.get("ok"):
                return
        except Exception:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for rollout server at {base_url}")


def wait_for_server_process(
    base_url: str,
    process: subprocess.Popen[bytes],
    log_path: str,
    timeout_s: float = 180.0,
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            log_text = ""
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
                    log_text = handle.read().strip()
            raise RuntimeError(
                f"rollout server exited early with code {process.returncode}.\n"
                f"log file: {log_path}\n{log_text}"
            )
        try:
            payload = request_json("GET", f"{base_url}/health", timeout=5.0)
            if payload.get("ok"):
                return
        except Exception:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for rollout server at {base_url}. log file: {log_path}")


def start_rollout_server(
    args: argparse.Namespace,
    model_path: str,
    output_dir: str,
    port: int,
) -> tuple[subprocess.Popen[bytes], str]:
    rollout_env = os.environ.copy()
    rollout_env["CUDA_VISIBLE_DEVICES"] = args.rollout_gpus
    for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]:
        rollout_env.pop(key, None)
    rollout_env.update(collect_patch_env(os.environ))
    log_dir = ensure_dir(Path(args.run_dir) / "server_logs")
    log_path = tempfile.NamedTemporaryFile(
        prefix="rollout_server_",
        suffix=".log",
        dir=log_dir,
        delete=False,
    ).name

    tp_size = len([gpu for gpu in args.rollout_gpus.split(",") if gpu.strip()])
    command = [
        sys.executable,
        "-u",
        "-m",
        "rl_torchtitan_vllm.rollout_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model-path",
        model_path,
        "--temp-checkpoint-dir",
        output_dir,
        "--tensor-parallel-size",
        str(tp_size),
        "--gpu-memory-utilization",
        str(args.vllm_gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
    ]
    log_handle = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        env=rollout_env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return process, log_path


def stop_rollout_server(base_url: str, process: subprocess.Popen[bytes]) -> None:
    try:
        request_json("POST", f"{base_url}/shutdown", {})
    except Exception:
        process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def run_rollout_phase(
    base_url: str,
    checkpoint_path: str,
    prompt_texts: list[str],
    expected_answers: list[str],
    reward_fn_name: str,
    step: int,
    args: argparse.Namespace,
    rollout_path: str,
) -> None:
    request_json("POST", f"{base_url}/prepare", {"checkpoint_path": checkpoint_path})

    batches: list[dict[str, Any]] = []
    for _ in range(args.num_rollout_batches):
        response = request_json(
            "POST",
            f"{base_url}/generate",
            {
                "prompt_texts": prompt_texts,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "n_samples_per_prompt": args.group_size,
            },
        )
        batches.append(response["result"])

    request_json("POST", f"{base_url}/unload", {})
    atomic_write_json(
        rollout_path,
        {
            "step": step,
            "prompt_texts": prompt_texts,
            "expected_answers": expected_answers,
            "reward_fn": reward_fn_name,
            "group_size": args.group_size,
            "batches": batches,
        },
    )


def run_train_phase(
    args: argparse.Namespace,
    model_path: str,
    input_checkpoint: str,
    output_checkpoint: str,
    rollout_path: str,
    metrics_path: str,
    step: int,
) -> dict[str, Any]:
    train_env = os.environ.copy()
    train_env["CUDA_VISIBLE_DEVICES"] = args.train_gpus
    train_env.update(collect_patch_env(os.environ))
    nproc_per_node = len([gpu for gpu in args.train_gpus.split(",") if gpu.strip()])
    master_port = find_free_port()
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_port={master_port}",
        "-m",
        "rl_torchtitan_vllm.train_worker",
        "--step",
        str(step),
        "--input-checkpoint",
        input_checkpoint,
        "--output-checkpoint",
        output_checkpoint,
        "--model-path",
        model_path,
        "--rollout-path",
        rollout_path,
        "--metrics-path",
        metrics_path,
        "--learning-rate",
        str(args.learning_rate),
        "--train-micro-batch-size",
        str(args.train_micro_batch_size),
        "--grpo-beta",
        str(args.grpo_beta),
        "--logdir",
        args.logdir,
    ]
    if args.use_stable_grpo:
        command.append("--use-stable-grpo")
    if args.use_vllm_compat or detect_vllm_compat_mode():
        command.append("--use-vllm-compat")

    subprocess.run(command, env=train_env, check=True)
    return load_json(metrics_path)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(args.run_dir)
    ensure_dir(args.logdir)

    if args.model_path and args.checkpoint_path:
        model_path = args.model_path
        current_checkpoint = args.checkpoint_path
    else:
        current_checkpoint, model_path = download_and_convert_model(
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
        )

    if args.use_real_dataset:
        prompt_texts, expected_answers = load_gsm8k_dataset(
            split="train",
            num_samples=args.num_dataset_samples,
        )
    else:
        prompt_texts, expected_answers = None, None

    if not prompt_texts or not expected_answers:
        prompt_texts, expected_answers = build_default_prompts()
        reward_fn_name = "trivial_reward_function"
        args.max_new_tokens = min(args.max_new_tokens, 20)
    else:
        reward_fn_name = "math_reward_function"

    port = args.vllm_port or find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    print(f"Using rollout server at {base_url}")
    print(f"Loaded {len(prompt_texts)} prompts with reward function {reward_fn_name}")
    print(f"Patch env: {format_patch_env(os.environ)}")

    for step in range(args.num_steps):
        print(f"\n=== Step {step + 1}/{args.num_steps} ===")
        rollout_dir = ensure_dir(Path(args.run_dir) / "rollouts")
        checkpoint_dir = ensure_dir(Path(args.run_dir) / "checkpoints")
        metrics_dir = ensure_dir(Path(args.run_dir) / "metrics")

        rollout_path = os.path.join(rollout_dir, f"step_{step:04d}.json")
        next_checkpoint = os.path.join(checkpoint_dir, f"policy_step_{step + 1:04d}.safetensors")
        metrics_path = os.path.join(metrics_dir, f"step_{step:04d}.json")

        server_process, server_log_path = start_rollout_server(args, model_path, args.output_dir, port)
        try:
            wait_for_server_process(base_url, server_process, server_log_path)
            run_rollout_phase(
                base_url=base_url,
                checkpoint_path=current_checkpoint,
                prompt_texts=prompt_texts,
                expected_answers=expected_answers,
                reward_fn_name=reward_fn_name,
                step=step,
                args=args,
                rollout_path=rollout_path,
            )
        finally:
            stop_rollout_server(base_url, server_process)

        metrics = run_train_phase(
            args=args,
            model_path=model_path,
            input_checkpoint=current_checkpoint,
            output_checkpoint=next_checkpoint,
            rollout_path=rollout_path,
            metrics_path=metrics_path,
            step=step,
        )
        current_checkpoint = next_checkpoint

        print(
            f"loss={metrics['loss']:.4f} "
            f"pg={metrics['pg_loss']:.4f} "
            f"kl={metrics['kl_div']:.6f} "
            f"entropy={metrics['entropy']:.4f} "
            f"reward={metrics['reward_mean']:+.3f} "
            f"reward_std={metrics['reward_std']:.3f} "
            f"adv={metrics['advantage_mean']:+.3f}/{metrics['advantage_std']:.3f} "
            f"samples={metrics['total_samples']}"
        )
        if metrics.get("train_micro_batch_size", 0) > 0:
            print(f"train_micro_batch_size={metrics['train_micro_batch_size']}")
        for batch in metrics.get("batches", []):
            print(
                f"  rollout_batch={batch['batch_idx']} "
                f"reward={batch['reward_mean']:+.3f} "
                f"kl={batch['training_rollout_kl_div']:.6f} "
                f"entropy={batch['entropy']:.4f}"
            )
        print(f"sample={metrics['sample_completion'][:120]}...")
        print(f"step_report={metrics_path}")

    print("\nTraining complete.")
    print(f"Latest checkpoint: {current_checkpoint}")
    print(f"TensorBoard: tensorboard --logdir={args.logdir}")


if __name__ == "__main__":
    main()
