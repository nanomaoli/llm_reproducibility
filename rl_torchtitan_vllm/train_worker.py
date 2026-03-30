from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from src.patch import apply_patches

apply_patches()



import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.utils.tensorboard import SummaryWriter

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))

from rl_torchtitan_vllm.common import (
    atomic_write_json,
    compute_grpo_advantages,
    compute_grpo_advantages_stable,
    compute_policy_gradient_loss_vllm,
    format_patch_env,
    load_json,
    load_policy_model,
    math_reward_function,
    normalize_rewards,
    trivial_reward_function,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One FSDP policy update from saved rollout data")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--input-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--rollout-path", required=True)
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--logdir", default="./outputs/rl_training")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--grpo-beta", type=float, default=0.1)
    parser.add_argument("--train-micro-batch-size", type=int, default=0)
    parser.add_argument("--use-stable-grpo", action="store_true")
    parser.add_argument("--use-vllm-compat", action="store_true")
    parser.add_argument("--kl-coef", type=float, default=0.1)
    return parser.parse_args()


def init_distributed() -> tuple[int, int, int]:
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def shard_indices(total_items: int, rank: int, world_size: int) -> list[int]:
    if total_items < world_size:
        return [rank] if rank < total_items else []

    shard_size = total_items // world_size
    if total_items % world_size != 0:
        raise ValueError(
            f"Total rollout items ({total_items}) must be divisible by world_size ({world_size}) "
            "or smaller than world_size."
        )

    start = rank * shard_size
    end = start + shard_size
    return list(range(start, end))


def select_batch_items(batch: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    return {
        "completions": [batch["completions"][idx] for idx in indices],
        "token_ids": [batch["token_ids"][idx] for idx in indices],
        "token_log_probs": [batch["token_log_probs"][idx] for idx in indices],
        "prompt_token_ids": [batch["prompt_token_ids"][idx] for idx in indices],
        "advantages": batch["advantages"][indices],
        "rewards": batch["rewards"][indices],
    }


def iter_micro_batches(batch: dict[str, Any], micro_batch_size: int) -> list[dict[str, Any]]:
    total_items = len(batch["completions"])
    if total_items == 0:
        return []

    if micro_batch_size <= 0 or micro_batch_size >= total_items:
        return [batch]

    micro_batches = []
    for start in range(0, total_items, micro_batch_size):
        end = min(start + micro_batch_size, total_items)
        micro_batches.append(
            {
                "completions": batch["completions"][start:end],
                "token_ids": batch["token_ids"][start:end],
                "token_log_probs": batch["token_log_probs"][start:end],
                "prompt_token_ids": batch["prompt_token_ids"][start:end],
                "advantages": batch["advantages"][start:end],
                "rewards": batch["rewards"][start:end],
            }
        )
    return micro_batches


def average_tensor(value: float | torch.Tensor, device: torch.device) -> float:
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value, device=device)
    tensor = tensor.detach().to(device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def weighted_average(value: float, weight: int, device: torch.device) -> float:
    weighted_sum = torch.tensor(value * weight, dtype=torch.float32, device=device)
    total_weight = torch.tensor(float(weight), dtype=torch.float32, device=device)
    dist.all_reduce(weighted_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_weight, op=dist.ReduceOp.SUM)
    if total_weight.item() == 0:
        return 0.0
    return float((weighted_sum / total_weight).item())


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = init_distributed()
    device = torch.device("cuda", local_rank)
    if rank == 0:
        print(f"train worker patch env: {format_patch_env(os.environ)}", flush=True)

    model = load_policy_model(
        checkpoint_path=args.input_checkpoint,
        model_path=args.model_path,
        use_vllm_compat=args.use_vllm_compat,
    ).to(device)
    model.train()

    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=None,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=args.learning_rate)

    rollout_bundle = load_json(args.rollout_path)
    reward_fn_name = rollout_bundle["reward_fn"]
    expected_answers = rollout_bundle["expected_answers"]
    group_size = int(rollout_bundle["group_size"])
    num_rollout_batches = len(rollout_bundle["batches"])
    sample_completion = rollout_bundle["batches"][0]["completions"][0]

    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    reward_means: list[float] = []
    reward_stds: list[float] = []
    advantage_means: list[float] = []
    advantage_stds: list[float] = []
    pg_losses: list[float] = []
    kl_divs: list[float] = []
    entropies: list[float] = []
    ratios: list[float] = []
    clipped_fracs: list[float] = []
    batch_reports: list[dict[str, Any]] = []

    for batch_idx, batch in enumerate(rollout_bundle["batches"]):
        if reward_fn_name == "math_reward_function":
            rewards = math_reward_function(batch["completions"], expected_answers, group_size)
        else:
            rewards = trivial_reward_function(batch["completions"], expected_answers, group_size)

        rewards_normalized, reward_mean, reward_std = normalize_rewards(rewards)
        if args.use_stable_grpo:
            advantages = compute_grpo_advantages_stable(rewards_normalized, group_size)
        else:
            advantages = compute_grpo_advantages(rewards_normalized, group_size, beta=args.grpo_beta)

        batch["rewards"] = rewards
        batch["advantages"] = advantages
        local_indices = shard_indices(len(batch["completions"]), rank, world_size)
        local_batch = select_batch_items(batch, local_indices)
        local_item_count = len(local_batch["completions"])

        batch_pg_loss = 0.0
        batch_kl_div = 0.0
        batch_entropy = 0.0
        batch_ratio_mean = 0.0
        batch_ratio_clipped_frac = 0.0

        if local_item_count > 0:
            for micro_batch in iter_micro_batches(local_batch, args.train_micro_batch_size):
                micro_batch_count = len(micro_batch["completions"])
                loss, loss_metrics = compute_policy_gradient_loss_vllm(
                    fsdp_model,
                    micro_batch["token_ids"],
                    micro_batch["token_log_probs"],
                    micro_batch["prompt_token_ids"],
                    micro_batch["advantages"],
                    kl_coef=args.kl_coef,
                )

                micro_weight = micro_batch_count / local_item_count
                scaled_loss = loss * micro_weight / num_rollout_batches
                scaled_loss.backward()
                total_loss += scaled_loss.detach()

                batch_pg_loss += loss_metrics["pg_loss"] * micro_weight
                batch_kl_div += loss_metrics["kl_div"] * micro_weight
                batch_entropy += loss_metrics["entropy"] * micro_weight
                batch_ratio_mean += loss_metrics["ratio_mean"] * micro_weight
                batch_ratio_clipped_frac += loss_metrics["ratio_clipped_frac"] * micro_weight

        global_batch_reward_mean = average_tensor(reward_mean, device)
        global_batch_reward_std = average_tensor(reward_std, device)
        global_batch_advantage_mean = average_tensor(float(advantages.mean().item()), device)
        global_batch_advantage_std = average_tensor(float(advantages.std().item()), device)
        global_batch_pg_loss = weighted_average(batch_pg_loss, local_item_count, device)
        global_batch_kl_div = weighted_average(batch_kl_div, local_item_count, device)
        global_batch_entropy = weighted_average(batch_entropy, local_item_count, device)
        global_batch_ratio_mean = weighted_average(batch_ratio_mean, local_item_count, device)
        global_batch_ratio_clipped_frac = weighted_average(
            batch_ratio_clipped_frac,
            local_item_count,
            device,
        )

        reward_means.append(global_batch_reward_mean)
        reward_stds.append(global_batch_reward_std)
        advantage_means.append(global_batch_advantage_mean)
        advantage_stds.append(global_batch_advantage_std)
        pg_losses.append(global_batch_pg_loss)
        kl_divs.append(global_batch_kl_div)
        entropies.append(global_batch_entropy)
        ratios.append(global_batch_ratio_mean)
        clipped_fracs.append(global_batch_ratio_clipped_frac)

        if rank == 0:
            batch_reports.append(
                {
                    "batch_idx": batch_idx,
                    "reward_mean": global_batch_reward_mean,
                    "reward_std": global_batch_reward_std,
                    "advantage_mean": global_batch_advantage_mean,
                    "advantage_std": global_batch_advantage_std,
                    "training_rollout_kl_div": global_batch_kl_div,
                    "pg_loss": global_batch_pg_loss,
                    "entropy": global_batch_entropy,
                    "ratio_mean": global_batch_ratio_mean,
                    "ratio_clipped_frac": global_batch_ratio_clipped_frac,
                    "sample_completion": batch["completions"][0] if batch["completions"] else "",
                    "rollout": {
                        "completions": batch["completions"],
                        "log_probs": batch.get("log_probs", []),
                        "token_ids": batch["token_ids"],
                        "token_log_probs": batch["token_log_probs"],
                        "prompt_token_ids": batch["prompt_token_ids"],
                    },
                    "rewards": [float(value) for value in rewards.tolist()],
                    "advantages": [float(value) for value in advantages.tolist()],
                }
            )

    if hasattr(fsdp_model, "clip_grad_norm_"):
        fsdp_model.clip_grad_norm_(1.0)
    else:
        torch.nn.utils.clip_grad_norm_(fsdp_model.parameters(), 1.0)
    optimizer.step()

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, save_policy):
        full_state = fsdp_model.state_dict()

    if rank == 0:
        Path(args.output_checkpoint).parent.mkdir(parents=True, exist_ok=True)
        save_file(full_state, args.output_checkpoint)

    dist.barrier()

    averaged_metrics = {
        "step": args.step,
        "loss": average_tensor(total_loss, device),
        "reward_mean": average_tensor(safe_mean(reward_means), device),
        "reward_std": average_tensor(safe_mean(reward_stds), device),
        "advantage_mean": average_tensor(safe_mean(advantage_means), device),
        "advantage_std": average_tensor(safe_mean(advantage_stds), device),
        "pg_loss": average_tensor(safe_mean(pg_losses), device),
        "kl_div": average_tensor(safe_mean(kl_divs), device),
        "entropy": average_tensor(safe_mean(entropies), device),
        "ratio_mean": average_tensor(safe_mean(ratios), device),
        "ratio_clipped_frac": average_tensor(safe_mean(clipped_fracs), device),
        "sample_completion": sample_completion,
        "total_samples": len(rollout_bundle["prompt_texts"]) * group_size * num_rollout_batches,
        "train_micro_batch_size": args.train_micro_batch_size,
        "reward_fn": reward_fn_name,
        "prompt_texts": rollout_bundle["prompt_texts"],
        "expected_answers": expected_answers,
        "group_size": group_size,
        "num_rollout_batches": num_rollout_batches,
        "batches": batch_reports if rank == 0 else [],
    }

    if rank == 0:
        writer = SummaryWriter(args.logdir)
        writer.add_scalar("rl/loss", averaged_metrics["loss"], args.step)
        writer.add_scalar("rl/pg_loss", averaged_metrics["pg_loss"], args.step)
        writer.add_scalar("rl/kl_div", averaged_metrics["kl_div"], args.step)
        writer.add_scalar("rl/entropy", averaged_metrics["entropy"], args.step)
        writer.add_scalar("rl/ratio_mean", averaged_metrics["ratio_mean"], args.step)
        writer.add_scalar("rl/ratio_clipped_frac", averaged_metrics["ratio_clipped_frac"], args.step)
        writer.add_scalar("rl/reward_mean", averaged_metrics["reward_mean"], args.step)
        writer.add_scalar("rl/reward_std", averaged_metrics["reward_std"], args.step)
        writer.add_scalar("rl/advantage_mean", averaged_metrics["advantage_mean"], args.step)
        writer.add_scalar("rl/advantage_std", averaged_metrics["advantage_std"], args.step)
        writer.add_scalar("rl/total_samples", averaged_metrics["total_samples"], args.step)
        writer.close()
        atomic_write_json(args.metrics_path, averaged_metrics)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
