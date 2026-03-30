from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

from src.patch import apply_patches

apply_patches()

from torchtitan.experiments.deterministic_vllm_rl.weights.converter import (
    torchtitan_to_vllm,
    vllm_to_torchtitan,
)
from torchtitan.experiments.deterministic_vllm_rl.weights_vllm_compat import (
    torchtitan_to_vllm_compat,
)
from torchtitan.models.qwen3.model.args import Qwen3ModelArgs


PATCH_ENV_VAR_NAMES = (
    "VLLM_ATTENTION_BACKEND",
    "VLLM_BATCH_INVARIANT",
    "VLLM_TP_INVARIANT",
    "ALIGN_TRAIN_INFERENCE",
)


def collect_patch_env(source_env: dict[str, str] | None = None) -> dict[str, str]:
    env = source_env if source_env is not None else os.environ
    return {
        key: env[key]
        for key in PATCH_ENV_VAR_NAMES
        if key in env and env[key] != ""
    }


def format_patch_env(source_env: dict[str, str] | None = None) -> str:
    patch_env = collect_patch_env(source_env)
    if not patch_env:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in patch_env.items())


def ensure_dir(path: str | os.PathLike[str]) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def atomic_write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        temp_path = handle.name
    os.replace(temp_path, path)


def atomic_save_safetensors(
    tensors: dict[str, torch.Tensor],
    path: str | os.PathLike[str],
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(
        suffix=path.suffix or ".safetensors",
        dir=path.parent,
        delete=False,
    ) as handle:
        temp_path = handle.name
    save_file(tensors, temp_path)
    os.replace(temp_path, path)


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def detect_vllm_compat_mode() -> bool:
    from src.utils import (
        batch_invariant_is_enabled,
        compatible_mode_is_enabled,
        tp_invariant_is_enabled,
    )

    return (
        batch_invariant_is_enabled()
        or tp_invariant_is_enabled()
        or compatible_mode_is_enabled()
    )


def _build_model_args(model_path: str) -> Qwen3ModelArgs:
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    return Qwen3ModelArgs(
        dim=hf_config.hidden_size,
        n_layers=hf_config.num_hidden_layers,
        n_heads=hf_config.num_attention_heads,
        n_kv_heads=hf_config.num_key_value_heads,
        vocab_size=hf_config.vocab_size,
        head_dim=getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        ),
        hidden_dim=hf_config.intermediate_size,
        norm_eps=hf_config.rms_norm_eps,
        rope_theta=hf_config.rope_theta,
        max_seq_len=getattr(hf_config, "max_position_embeddings", 32768),
        qk_norm=True,
        depth_init=True,
        eos_id=getattr(hf_config, "eos_token_id", 151645),
    )


def _looks_like_vllm_compat_state(state_dict: dict[str, torch.Tensor]) -> bool:
    return any("gate_up_proj" in key or "down_proj" in key for key in state_dict)


def download_and_convert_model(
    model_name: str,
    cache_dir: str = "./models",
    output_dir: str = "./converted",
) -> tuple[str, str]:
    ensure_dir(output_dir)
    print(f"Downloading {model_name} from HuggingFace...")
    model_path = snapshot_download(
        model_name,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer.model"],
    )
    print(f"  Downloaded to: {model_path}")

    print("Converting weights to TorchTitan format...")
    titan_state = vllm_to_torchtitan(model_path)
    titan_checkpoint_path = os.path.join(output_dir, "qwen3_torchtitan.safetensors")
    atomic_save_safetensors(titan_state, titan_checkpoint_path)
    print(f"  Saved TorchTitan weights to: {titan_checkpoint_path}")
    return titan_checkpoint_path, model_path


def load_policy_model(
    checkpoint_path: str,
    model_path: str,
    use_vllm_compat: bool = True,
) -> torch.nn.Module:
    state_dict = load_file(checkpoint_path)
    model_args = _build_model_args(model_path)

    if use_vllm_compat:
        from torchtitan.experiments.deterministic_vllm_rl.models.qwen3 import (
            Qwen3VLLMCompatModel,
        )

        model = Qwen3VLLMCompatModel(model_args)
        if not _looks_like_vllm_compat_state(state_dict):
            state_dict = torchtitan_to_vllm_compat(state_dict)
        model.load_state_dict(state_dict, strict=False)
    else:
        if _looks_like_vllm_compat_state(state_dict):
            raise ValueError(
                "Checkpoint is already in vLLM-compat format. "
                "Load with use_vllm_compat=True."
            )
        from torchtitan.models.qwen3 import Qwen3Model

        model = Qwen3Model(model_args)
        model.load_state_dict(state_dict, strict=False)

    model.to(torch.bfloat16)
    return model


def prepare_vllm_model_dir(base_model_path: str, temp_model_dir: str) -> None:
    ensure_dir(temp_model_dir)
    for file_name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "merges.txt",
        "vocab.json",
        "tokenizer.model",
    ]:
        source = os.path.join(base_model_path, file_name)
        if os.path.exists(source):
            shutil.copy2(source, os.path.join(temp_model_dir, file_name))

    for shard_file in Path(base_model_path).glob("model-*.safetensors"):
        shutil.copy2(shard_file, Path(temp_model_dir) / shard_file.name)

    index_file = os.path.join(base_model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        shutil.copy2(index_file, os.path.join(temp_model_dir, "model.safetensors.index.json"))


def write_vllm_weights(
    checkpoint_path: str,
    temp_model_dir: str,
) -> None:
    state_dict = load_file(checkpoint_path)
    vllm_state = torchtitan_to_vllm(state_dict)
    checkpoint_output_path = os.path.join(temp_model_dir, "model.safetensors")
    shard_files = sorted(str(path) for path in Path(temp_model_dir).glob("model-*.safetensors"))
    index_file = os.path.join(temp_model_dir, "model.safetensors.index.json")

    if len(shard_files) == 2 and os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as handle:
            index_data = json.load(handle)
        weight_map = index_data["weight_map"]

        shard_payloads = {shard_files[0]: {}, shard_files[1]: {}}
        for key, value in vllm_state.items():
            target_shard = weight_map.get(key, os.path.basename(shard_files[0]))
            if target_shard.endswith("model-00001-of-00002.safetensors"):
                shard_payloads[shard_files[0]][key] = value
            else:
                shard_payloads[shard_files[1]][key] = value

        for shard_file, payload in shard_payloads.items():
            cast_payload = {
                key: tensor.to(torch.bfloat16) if tensor.dtype == torch.float32 else tensor
                for key, tensor in payload.items()
            }
            atomic_save_safetensors(cast_payload, shard_file)
        return

    cast_state = {
        key: tensor.to(torch.bfloat16) if tensor.dtype == torch.float32 else tensor
        for key, tensor in vllm_state.items()
    }
    atomic_save_safetensors(cast_state, checkpoint_output_path)


def extract_numeric_answer(text: str) -> str | None:
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")

    numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def load_gsm8k_dataset(
    split: str = "train",
    num_samples: int = 100,
    offset: int = 0,
) -> tuple[list[str] | None, list[str] | None]:
    try:
        from datasets import load_dataset

        dataset = load_dataset("openai/gsm8k", "main", split=split)
        prompts: list[str] = []
        answers: list[str] = []

        skipped = 0
        for item in dataset:
            if skipped < offset:
                skipped += 1
                continue
            if len(prompts) >= num_samples:
                break

            answer_num = extract_numeric_answer(item["answer"])
            if answer_num is None:
                continue

            prompts.append(f"Question: {item['question']}\nAnswer:")
            answers.append(answer_num)

        return prompts, answers
    except ImportError:
        print("datasets is not installed. Falling back to default prompts.")
        return None, None
    except Exception as exc:
        print(f"Failed to load GSM8K dataset: {exc}")
        return None, None


def build_default_prompts() -> tuple[list[str], list[str]]:
    prompts_with_answers = [
        ("The capital of France is", "paris"),
        ("What is 7 times 8?", "56"),
        ("The first president of the United States was", "washington"),
        ("The chemical symbol for water is", "h2o"),
        ("The largest planet in our solar system is", "jupiter"),
    ]
    return [item[0] for item in prompts_with_answers], [item[1] for item in prompts_with_answers]


def math_reward_function(
    completions: list[str],
    expected_answers: list[str],
    group_size: int = 4,
) -> torch.Tensor:
    rewards = []
    for idx, completion in enumerate(completions):
        prompt_idx = idx // group_size
        expected = expected_answers[prompt_idx].strip().lower()
        predicted = extract_numeric_answer(completion)
        rewards.append(1.0 if predicted is not None and predicted.lower() == expected else 0.0)
    return torch.tensor(rewards, dtype=torch.float32)


def trivial_reward_function(
    completions: list[str],
    expected_answers: list[str] | None = None,
    group_size: int = 4,
) -> torch.Tensor:
    rewards = []
    for idx, completion in enumerate(completions):
        reward = 1.0
        if not completion:
            rewards.append(0.0)
            continue

        total_chars = len(completion)
        non_ascii_ratio = sum(1 for char in completion if ord(char) > 127) / total_chars
        if non_ascii_ratio > 0.1:
            reward *= 0.1

        uppercase_ratio = sum(1 for char in completion if char.isupper()) / total_chars
        reward *= 1.0 - 0.9 * uppercase_ratio

        if expected_answers is not None:
            prompt_idx = idx // group_size
            expected = expected_answers[prompt_idx].lower()
            reward *= 2.0 if expected in completion.lower() else 0.5

        rewards.append(reward)
    return torch.tensor(rewards, dtype=torch.float32)


def normalize_rewards(rewards: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    reward_mean = rewards.mean()
    reward_std = rewards.std()
    if reward_std > 1e-8:
        normalized = (rewards - reward_mean) / reward_std
    else:
        normalized = rewards - reward_mean
    return normalized, reward_mean.item(), reward_std.item()


def compute_mc_pass_at_1(
    completions: list[str],
    expected_answers: list[str],
    num_eval_per_sample: int,
) -> dict[str, Any]:
    rewards = math_reward_function(
        completions,
        expected_answers,
        group_size=num_eval_per_sample,
    )
    grouped = rewards.view(len(expected_answers), num_eval_per_sample)
    per_prompt_pass_at_1 = grouped.mean(dim=1)
    any_correct_rate = (grouped.max(dim=1).values > 0).float().mean().item()
    pass_at_1 = per_prompt_pass_at_1.mean().item()
    if len(expected_answers) > 1:
        stderr = (
            per_prompt_pass_at_1.std(unbiased=True).item()
            / math.sqrt(len(expected_answers))
        )
    else:
        stderr = 0.0

    return {
        "pass_at_1": pass_at_1,
        "pass_at_1_stderr": stderr,
        "any_correct_rate": any_correct_rate,
        "per_prompt_pass_at_1": [float(value) for value in per_prompt_pass_at_1.tolist()],
        "rewards": [float(value) for value in rewards.tolist()],
    }


def compute_grpo_advantages(
    rewards: torch.Tensor,
    group_size: int = 4,
    beta: float = 0.1,
) -> torch.Tensor:
    batch_size = rewards.shape[0]
    if batch_size % group_size != 0:
        raise ValueError(f"Batch size {batch_size} must be divisible by group_size {group_size}")

    grouped = rewards.view(batch_size // group_size, group_size)
    exp_rewards = torch.exp(grouped / beta)
    return (exp_rewards / exp_rewards.mean(dim=1, keepdim=True) - 1.0).view(-1)


def compute_grpo_advantages_stable(rewards: torch.Tensor, group_size: int = 4) -> torch.Tensor:
    batch_size = rewards.shape[0]
    if batch_size % group_size != 0:
        raise ValueError(f"Batch size {batch_size} must be divisible by group_size {group_size}")

    grouped = rewards.view(batch_size // group_size, group_size)
    return (grouped - grouped.mean(dim=1, keepdim=True)).view(-1)


def compute_policy_gradient_loss_vllm(
    model: torch.nn.Module,
    vllm_token_ids: list[list[int]],
    vllm_token_log_probs: list[list[float]],
    prompt_token_ids: list[list[int]],
    advantages: torch.Tensor,
    kl_coef: float = 0.1,
    ppo_clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
) -> tuple[torch.Tensor, dict[str, Any]]:
    device = next(model.parameters()).device
    advantages = advantages.to(device)
    ref_log_probs = torch.stack(
        [torch.tensor(token_lps, dtype=torch.float32, device=device).sum() for token_lps in vllm_token_log_probs]
    )

    batch_token_log_probs: list[torch.Tensor] = []
    batch_total_log_probs: list[torch.Tensor] = []
    first_sample_deltas: list[dict[str, Any]] = []

    for idx, (prompt_toks, gen_toks, ref_token_lps) in enumerate(
        zip(prompt_token_ids, vllm_token_ids, vllm_token_log_probs)
    ):
        if not gen_toks:
            empty = torch.zeros(0, dtype=torch.float32, device=device)
            batch_token_log_probs.append(empty)
            batch_total_log_probs.append(torch.zeros((), dtype=torch.float32, device=device))
            continue

        full_sequence = prompt_toks + gen_toks
        full_tensor = torch.tensor(full_sequence, dtype=torch.long, device=device).unsqueeze(0)
        batch_size, seq_len = full_tensor.shape
        padded_len = math.ceil(seq_len / 16) * 16
        pad_len = padded_len - seq_len

        if pad_len > 0:
            pad_tensor = torch.zeros(
                (batch_size, pad_len),
                dtype=torch.long,
                device=full_tensor.device,
            )
            full_tensor_padded = torch.cat([full_tensor, pad_tensor], dim=1)
        else:
            full_tensor_padded = full_tensor

        # Keep the original simple_rl.py padding path for train/inference alignment.
        attention_mask = torch.ones(
            (batch_size, padded_len),
            dtype=torch.long,
            device=full_tensor.device,
        )
        if pad_len > 0:
            attention_mask[:, :-pad_len] = 0

        logits = model(full_tensor_padded, attention_masks=attention_mask)
        if pad_len > 0:
            log_probs = F.log_softmax(
                logits[:, :-1 - pad_len, :].to(torch.float32),
                dim=-1,
            )
            target_tokens = full_tensor_padded[:, 1:-pad_len]
        else:
            log_probs = F.log_softmax(logits[:, :-1, :].to(torch.float32), dim=-1)
            target_tokens = full_tensor_padded[:, 1:]

        prompt_len = len(prompt_toks)
        gen_start_idx = prompt_len - 1
        gen_end_idx = gen_start_idx + len(gen_toks)

        token_lps = log_probs[0, gen_start_idx:gen_end_idx, :].gather(
            1,
            target_tokens[0, gen_start_idx:gen_end_idx].unsqueeze(-1),
        ).squeeze(-1)

        batch_token_log_probs.append(token_lps)
        batch_total_log_probs.append(token_lps.sum())

        if idx == 0:
            titan_lps = token_lps.detach().cpu().float()
            for token_id, ref_lp, titan_lp in zip(gen_toks, ref_token_lps, titan_lps):
                first_sample_deltas.append(
                    {
                        "token_id": int(token_id),
                        "vllm_logprob": float(ref_lp),
                        "titan_logprob_f32": float(titan_lp.item()),
                    }
                )

    total_log_probs = torch.stack(batch_total_log_probs)
    log_ratio = total_log_probs - ref_log_probs
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1 - ppo_clip_eps, 1 + ppo_clip_eps)
    clipped = clipped_ratio * advantages
    pg_loss = -torch.min(unclipped, clipped).mean()

    non_empty_log_probs = [tensor for tensor in batch_token_log_probs if tensor.numel() > 0]
    if non_empty_log_probs:
        entropy = -torch.cat(non_empty_log_probs).mean()
    else:
        entropy = torch.zeros((), dtype=torch.float32, device=device)

    entropy_bonus = -entropy_coef * entropy
    kl_div = (ratio - 1 - log_ratio).mean()
    total_loss = pg_loss + entropy_bonus + kl_coef * kl_div

    metrics = {
        "pg_loss": float(pg_loss.item()),
        "entropy": float(entropy.item()),
        "kl_div": float(kl_div.item()),
        "ratio_mean": float(ratio.mean().item()),
        "ratio_clipped_frac": float((torch.abs(ratio - clipped_ratio) > 1e-6).float().mean().item()),
        "per_token_deltas": first_sample_deltas,
    }
    return total_loss, metrics
