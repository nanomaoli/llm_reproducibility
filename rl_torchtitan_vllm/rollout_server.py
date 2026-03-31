from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch
from transformers import AutoTokenizer

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.patch import apply_patches

apply_patches()

from rl_torchtitan_vllm.common import (
    ensure_dir,
    format_patch_env,
    prepare_vllm_model_dir,
    write_vllm_weights,
)




class RolloutEngine:
    def __init__(
        self,
        model_path: str,
        temp_checkpoint_dir: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
    ) -> None:
        self.base_model_path = model_path
        self.temp_model_dir = os.path.abspath(
            os.path.join(temp_checkpoint_dir, f"vllm_temp_model_{os.getpid()}")
        )
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.llm: Any | None = None
        self.loaded_checkpoint: str | None = None
        self._sampling_params_cls: Any | None = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
        )
        if os.path.exists(self.temp_model_dir):
            shutil.rmtree(self.temp_model_dir)
        prepare_vllm_model_dir(self.base_model_path, self.temp_model_dir)

    def _ensure_vllm_imported(self) -> None:
        if self._sampling_params_cls is not None:
            return
        from vllm import LLM, SamplingParams

        self._llm_cls = LLM
        self._sampling_params_cls = SamplingParams

    def prepare(self, checkpoint_path: str) -> None:
        if self.loaded_checkpoint == checkpoint_path and self.llm is not None:
            return

        self._ensure_vllm_imported()
        if not os.path.exists(self.temp_model_dir):
            prepare_vllm_model_dir(self.base_model_path, self.temp_model_dir)
        write_vllm_weights(checkpoint_path, self.temp_model_dir)
        if self.llm is None:
            self.llm = self._llm_cls(
                model=self.temp_model_dir,
                trust_remote_code=False,
                max_model_len=self.max_model_len,
                dtype="bfloat16",
                gpu_memory_utilization=self.gpu_memory_utilization,
                tensor_parallel_size=self.tensor_parallel_size,
                seed=42,
                enable_prefix_caching=False,
                max_num_batched_tokens=10240,
                enforce_eager=True,
            )
        else:
            self.llm.collective_rpc("reload_weights")
        self.loaded_checkpoint = checkpoint_path

    def cleanup_model_dir(self) -> None:
        if os.path.exists(self.temp_model_dir):
            shutil.rmtree(self.temp_model_dir)

    def generate(
        self,
        prompt_texts: list[str],
        max_new_tokens: int,
        temperature: float,
        n_samples_per_prompt: int,
    ) -> dict[str, Any]:
        if self.llm is None:
            raise RuntimeError("Model weights are not loaded. Call /prepare first.")

        sampling_params = self._sampling_params_cls(
            temperature=temperature,
            max_tokens=max_new_tokens,
            n=n_samples_per_prompt,
            seed=42,
            logprobs=1,
        )
        outputs = self.llm.generate(prompt_texts, sampling_params)

        completions: list[str] = []
        log_probs: list[float] = []
        token_ids: list[list[int]] = []
        token_log_probs: list[list[float]] = []
        prompt_token_ids: list[list[int]] = []

        for prompt_text, output in zip(prompt_texts, outputs):
            prompt_ids = getattr(output, "prompt_token_ids", None)
            if prompt_ids is None:
                prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_ids = [int(token_id) for token_id in prompt_ids]
            for sample in output.outputs:
                per_token = [
                    float(list(logprob_dict.values())[0].logprob)
                    for logprob_dict in sample.logprobs
                ]
                completions.append(sample.text)
                log_probs.append(float(sum(per_token)))
                token_ids.append([int(token_id) for token_id in sample.token_ids])
                token_log_probs.append(per_token)
                prompt_token_ids.append(prompt_ids)

        return {
            "completions": completions,
            "log_probs": log_probs,
            "token_ids": token_ids,
            "token_log_probs": token_log_probs,
            "prompt_token_ids": prompt_token_ids,
        }

    def unload(self, remove_model_dir: bool = False) -> None:
        if self.llm is not None:
            executor = getattr(getattr(self.llm, "llm_engine", None), "model_executor", None)
            if executor is not None and hasattr(executor, "shutdown"):
                try:
                    executor.shutdown()
                except Exception:
                    pass

            self.llm = None
            self.loaded_checkpoint = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if remove_model_dir:
            self.cleanup_model_dir()


class JsonHandler(BaseHTTPRequestHandler):
    engine: RolloutEngine
    engine_lock: threading.Lock

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length else b"{}"
        return json.loads(body.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        self._send_json(
            200,
            {
                "ok": True,
                "loaded_checkpoint": self.engine.loaded_checkpoint,
                "engine_loaded": self.engine.llm is not None,
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_json_body()
            with self.engine_lock:
                if self.path == "/prepare":
                    self.engine.prepare(payload["checkpoint_path"])
                    self._send_json(200, {"ok": True})
                    return

                if self.path == "/generate":
                    result = self.engine.generate(
                        prompt_texts=payload["prompt_texts"],
                        max_new_tokens=int(payload["max_new_tokens"]),
                        temperature=float(payload["temperature"]),
                        n_samples_per_prompt=int(payload["n_samples_per_prompt"]),
                    )
                    self._send_json(200, {"ok": True, "result": result})
                    return

                if self.path == "/unload":
                    self.engine.unload()
                    self._send_json(200, {"ok": True})
                    return

                if self.path == "/shutdown":
                    self.engine.unload(remove_model_dir=True)
                    self._send_json(200, {"ok": True})
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                    return

            self._send_json(404, {"ok": False, "error": "not found"})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        return


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone vLLM rollout server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=28600)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--temp-checkpoint-dir", default="./converted")
    parser.add_argument("--tensor-parallel-size", type=int, default=torch.cuda.device_count())
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--max-model-len", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.temp_checkpoint_dir)

    for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]:
        os.environ.pop(key, None)

    print(f"rollout server patch env: {format_patch_env(os.environ)}", flush=True)

    JsonHandler.engine = RolloutEngine(
        model_path=args.model_path,
        temp_checkpoint_dir=args.temp_checkpoint_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    JsonHandler.engine_lock = threading.Lock()

    server = ReusableThreadingHTTPServer((args.host, args.port), JsonHandler)
    print(f"rollout server listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        JsonHandler.engine.unload(remove_model_dir=True)


if __name__ == "__main__":
    main()
