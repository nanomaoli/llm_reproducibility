from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import tempfile
import json
import math
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from rl_torchtitan_vllm.common import (
    atomic_write_json,
    build_default_prompts,
    compute_mc_pass_at_1,
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
    parser.add_argument("--rollout-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--train-micro-batch-size", type=int, default=0)
    parser.add_argument("--grpo-beta", type=float, default=0.1)
    parser.add_argument("--use-stable-grpo", action="store_true")
    parser.add_argument("--use-real-dataset", action="store_true")
    parser.add_argument("--use-vllm-compat", action="store_true")
    parser.add_argument("--num-train-samples", type=int, default=16)
    parser.add_argument("--num-test-samples", type=int, default=16)
    parser.add_argument("--eval-every-n-steps", type=int, default=0)
    parser.add_argument("--num-eval-per-sample", type=int, default=16)
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
    log_handle.close()
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


def select_prompt_batch(
    prompts: list[str],
    answers: list[str],
    start_idx: int,
    batch_size: int,
) -> tuple[list[str], list[str], int]:
    if not prompts:
        return [], [], start_idx

    batch_size = max(1, min(batch_size, len(prompts)))
    indices = [(start_idx + offset) % len(prompts) for offset in range(batch_size)]
    next_idx = (start_idx + batch_size) % len(prompts)
    return [prompts[idx] for idx in indices], [answers[idx] for idx in indices], next_idx


def build_prompt_groups(
    prompts: list[str],
    expected_answers: list[str],
    completions: list[str],
    rewards: list[float],
    group_size: int,
) -> list[dict[str, Any]]:
    groups = []
    for prompt_idx, (prompt, answer) in enumerate(zip(prompts, expected_answers)):
        start = prompt_idx * group_size
        end = start + group_size
        groups.append(
            {
                "prompt": prompt,
                "expected_answer": answer,
                "completions": completions[start:end],
                "rewards": rewards[start:end],
            }
        )
    return groups


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

    request_json("POST", f"{base_url}/unload", {})
    atomic_write_json(
        rollout_path,
        {
            "step": step,
            "prompt_texts": prompt_texts,
            "expected_answers": expected_answers,
            "reward_fn": reward_fn_name,
            "group_size": args.group_size,
            "rollout_batch_size": len(prompt_texts),
            "rollout": response["result"],
        },
    )


def run_eval_phase(
    base_url: str,
    checkpoint_path: str,
    prompt_texts: list[str],
    expected_answers: list[str],
    args: argparse.Namespace,
    eval_path: str,
    step: int,
) -> dict[str, Any]:
    request_json("POST", f"{base_url}/prepare", {"checkpoint_path": checkpoint_path})
    response = request_json(
        "POST",
        f"{base_url}/generate",
        {
            "prompt_texts": prompt_texts,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "n_samples_per_prompt": args.num_eval_per_sample,
        },
        timeout=1800.0,
    )
    request_json("POST", f"{base_url}/unload", {})

    result = response["result"]
    eval_metrics = compute_mc_pass_at_1(
        completions=result["completions"],
        expected_answers=expected_answers,
        num_eval_per_sample=args.num_eval_per_sample,
    )
    prompt_groups = build_prompt_groups(
        prompts=prompt_texts,
        expected_answers=expected_answers,
        completions=result["completions"],
        rewards=eval_metrics["rewards"],
        group_size=args.num_eval_per_sample,
    )
    report = {
        "step": step,
        "num_test_samples": len(prompt_texts),
        "num_eval_per_sample": args.num_eval_per_sample,
        "pass_at_1": eval_metrics["pass_at_1"],
        "pass_at_1_stderr": eval_metrics["pass_at_1_stderr"],
        "any_correct_rate": eval_metrics["any_correct_rate"],
        "per_prompt_pass_at_1": eval_metrics["per_prompt_pass_at_1"],
        "prompt_groups": prompt_groups,
        "rollout": result,
    }
    atomic_write_json(eval_path, report)
    return report


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

    writer = SummaryWriter(args.logdir)

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
        train_prompt_texts, train_expected_answers = load_gsm8k_dataset(
            split="train",
            num_samples=args.num_train_samples,
        )
        test_prompt_texts, test_expected_answers = load_gsm8k_dataset(
            split="test",
            num_samples=args.num_test_samples,
        )
    else:
        train_prompt_texts, train_expected_answers = None, None
        test_prompt_texts, test_expected_answers = None, None

    if not train_prompt_texts or not train_expected_answers:
        fallback_prompts, fallback_answers = build_default_prompts()
        train_size = min(args.num_train_samples, len(fallback_prompts))
        test_size = min(args.num_test_samples, max(1, len(fallback_prompts) - train_size))
        train_prompt_texts = fallback_prompts[:train_size]
        train_expected_answers = fallback_answers[:train_size]
        test_prompt_texts = fallback_prompts[-test_size:]
        test_expected_answers = fallback_answers[-test_size:]
        reward_fn_name = "trivial_reward_function"
        args.max_new_tokens = min(args.max_new_tokens, 20)
    else:
        reward_fn_name = "math_reward_function"
        if not test_prompt_texts or not test_expected_answers:
            test_prompt_texts = train_prompt_texts[: min(args.num_test_samples, len(train_prompt_texts))]
            test_expected_answers = train_expected_answers[: len(test_prompt_texts)]

    port = args.vllm_port or find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    print(f"Using rollout server at {base_url}")
    print(
        f"Loaded {len(train_prompt_texts)} train prompts and "
        f"{len(test_prompt_texts)} test prompts with reward function {reward_fn_name}"
    )
    print(f"Patch env: {format_patch_env(os.environ)}")
    rollout_batch_size = max(1, min(args.rollout_batch_size, len(train_prompt_texts)))
    rollout_cursor = 0
    current_policy_checkpoint = os.path.join(args.run_dir, "current_policy.safetensors")

    for step in tqdm(list(range(args.num_steps))):
        print(f"\n=== Step {step + 1}/{args.num_steps} ===")
        rollout_dir = ensure_dir(Path(args.run_dir) / "rollouts")
        checkpoint_dir = ensure_dir(Path(args.run_dir) / "checkpoints")
        metrics_dir = ensure_dir(Path(args.run_dir) / "metrics")
        eval_dir = ensure_dir(Path(args.run_dir) / "evals")

        rollout_path = os.path.join(rollout_dir, f"step_{step:04d}.json")
        metrics_path = os.path.join(metrics_dir, f"step_{step:04d}.json")
        eval_path = os.path.join(eval_dir, f"step_{step:04d}.json")
        rollout_prompts, rollout_answers, rollout_cursor = select_prompt_batch(
            train_prompt_texts,
            train_expected_answers,
            rollout_cursor,
            rollout_batch_size,
        )

        server_process, server_log_path = start_rollout_server(args, model_path, args.output_dir, port)
        try:
            wait_for_server_process(base_url, server_process, server_log_path)
            run_rollout_phase(
                base_url=base_url,
                checkpoint_path=current_checkpoint,
                prompt_texts=rollout_prompts,
                expected_answers=rollout_answers,
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
            output_checkpoint=current_policy_checkpoint,
            rollout_path=rollout_path,
            metrics_path=metrics_path,
            step=step,
        )
        current_checkpoint = current_policy_checkpoint

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
        rollout_report = metrics.get("rollout")
        if rollout_report:
            print(
                f"  rollout reward={rollout_report['reward_mean']:+.3f} "
                f"kl={rollout_report['training_rollout_kl_div']:.6f} "
                f"entropy={rollout_report['entropy']:.4f}"
            )
        print(f"sample={metrics['sample_completion'][:120]}...")
        print(f"step_report={metrics_path}")

        should_eval = (
            args.eval_every_n_steps > 0
            and (step + 1) % args.eval_every_n_steps == 0
            and reward_fn_name == "math_reward_function"
            and len(test_prompt_texts) > 0
        )
        if should_eval:
            eval_server_process, eval_server_log_path = start_rollout_server(
                args,
                model_path,
                args.output_dir,
                port,
            )
            try:
                wait_for_server_process(base_url, eval_server_process, eval_server_log_path)
                eval_report = run_eval_phase(
                    base_url=base_url,
                    checkpoint_path=current_checkpoint,
                    prompt_texts=test_prompt_texts,
                    expected_answers=test_expected_answers,
                    args=args,
                    eval_path=eval_path,
                    step=step,
                )
            finally:
                stop_rollout_server(base_url, eval_server_process)

            archived_checkpoint = os.path.join(
                checkpoint_dir,
                f"eval_step_{step + 1:04d}.safetensors",
            )
            shutil.copy2(current_checkpoint, archived_checkpoint)
            writer.add_scalar("eval/pass_at_1", eval_report["pass_at_1"], step)
            writer.add_scalar("eval/pass_at_1_stderr", eval_report["pass_at_1_stderr"], step)
            writer.add_scalar("eval/any_correct_rate", eval_report["any_correct_rate"], step)
            print(
                f"eval_pass@1={eval_report['pass_at_1']:.4f} "
                f"+/- {eval_report['pass_at_1_stderr']:.4f} "
                f"any_correct={eval_report['any_correct_rate']:.4f}"
            )
            print(f"eval_report={eval_path}")
            print(f"saved_checkpoint={archived_checkpoint}")

    print("\nTraining complete.")
    print(f"Latest checkpoint: {current_checkpoint}")
    print(f"TensorBoard: tensorboard --logdir={args.logdir}")
    writer.close()


if __name__ == "__main__":
    main()
