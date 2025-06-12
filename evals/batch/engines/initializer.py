"""Engine initializers.
Note that this file should not import any engine dependent modeules, such as
vLLM, because the engine initializer is used in the driver node which may
not have GPUs.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from ..utils import (
    download_model_from_hf,
    update_dict_recursive,
)
from ..workload import EvalWorkload
from .base import EngineBase


class EngineInitializerBase:
    """Base class for engine initializer.

    Args:
        model_id: The model id.
        accelerator_type: The accelerator type.
        engine_kwargs: The engine specific configurations.
        ray_env_vars: The Ray runtime environment
    """

    use_ray_placement_group: bool = False

    def __init__(
        self,
        model_id: str,
        accelerator_type: str,
        engine_kwargs: Dict[str, Any],
        lora_adapter: Optional[str] = None,
        ray_env_vars: Dict[str, Any] = None,
    ):
        self._model = model_id
        self._accelerator_type = accelerator_type
        self._ray_env_vars = ray_env_vars or {}
        self.lora_adapter = lora_adapter
        self.engine_kwargs = engine_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def accelerator_type(self) -> str:
        return self._accelerator_type

    @property
    def ray_env_vars(self) -> Dict[str, str]:
        return self._ray_env_vars

    @property
    def num_gpus(self) -> int:
        """The number of GPUs used per engine."""
        raise NotImplementedError

    @property
    def max_model_len(self) -> Optional[int]:
        """The maximum model length set by the engine."""
        return None

    def get_engine_cls(self) -> EngineBase:
        """Get the engine class.

        Returns:
            The engine class.
        """
        raise NotImplementedError

    def get_engine_constructor_args(self, workload: EvalWorkload) -> Dict[str, Any]:
        """Get the engine constructor arguments.

        Args:
            workload: The workload that the engine will process.

        Returns:
            The engine constructor keyword arguments.
        """
        raise NotImplementedError


class vLLMEngineInitializer(EngineInitializerBase):
    use_ray_placement_group: bool = False

    def __init__(
        self,
        model_id: str,
        accelerator_type: str,
        engine_kwargs: Dict[str, Any],
        lora_adapter: Optional[str] = None,
        ray_env_vars: Dict[str, Any] = None,
    ):
        super().__init__(
            model_id, accelerator_type, engine_kwargs, lora_adapter, ray_env_vars
        )

        # Override vLLM default configs. Note that this is only effective
        # when the config is not set by users.
        self.engine_kwargs.setdefault("gpu_memory_utilization", 0.95)
        self.engine_kwargs.setdefault("use_v2_block_manager", True)
        self.engine_kwargs.setdefault("enable_prefix_caching", False)
        self.engine_kwargs.setdefault("enforce_eager", False)
        self.engine_kwargs.setdefault("pipeline_parallel_size", 1)
        self.engine_kwargs.setdefault("max_num_seqs", 256)
        self.engine_kwargs.setdefault("tensor_parallel_size", 1)
        self.engine_kwargs.setdefault("max_logprobs", 0)
        self.engine_kwargs.setdefault("distributed_executor_backend", "mp")

        # Set engine environment variables.
        self._ray_env_vars.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
        self._ray_env_vars.setdefault("ENABLE_ANYSCALE_PREFIX_OPTIMIZATIONS", "0")
        # FIXME: This should already be deprecated and can be removed.
        self._ray_env_vars.setdefault("VLLM_DISABLE_LOGPROBS", "1")
        for key, value in self._ray_env_vars.items():
            os.environ[key] = str(value)

    def get_engine_cls(self):
        from .vllm_engine import AsyncLLMPredictor

        return AsyncLLMPredictor

    @property
    def num_gpus(self) -> int:
        assert "tensor_parallel_size" in self.engine_kwargs
        assert "pipeline_parallel_size" in self.engine_kwargs
        tp_size = self.engine_kwargs["tensor_parallel_size"]
        pp_size = self.engine_kwargs["pipeline_parallel_size"]
        return tp_size * pp_size

    @property
    def max_model_len(self) -> Optional[int]:
        """The maximum model length set by the engine."""
        return self.engine_kwargs.get("max_model_len", None)

    def get_engine_constructor_args(self, workload: EvalWorkload):
        from vllm import PoolingParams, SamplingParams
        from vllm.config import PoolerConfig

        constructor_kwargs = {
            "model": self.model,
            "lora_adapter": self.lora_adapter,
        }

        if sampling_params := workload.sampling_params:
            # Sampling params is given: Auto-regressive generation.
            # In this case, we need to set max_tokens and max_model_len.

            max_tokens = sampling_params.get("max_tokens", None)
            if max_tokens is None:
                raise ValueError("max_tokens is required for vLLM engine.")

            vllm_sampling_params = SamplingParams(**workload.sampling_params)
            vllm_sampling_params.max_tokens = max_tokens
            vllm_sampling_params.detokenize = False
            constructor_kwargs["params"] = vllm_sampling_params

            if (
                "max_model_len" not in self.engine_kwargs
                and workload.max_tokens_in_prompt < 0
            ):
                raise ValueError(
                    "Neither max_tokens_in_prompt nor max_model_len is set. If you "
                    "intend to let the pipeline infer max_tokens_in_prompt but got this error, "
                    "it is either because the workload has not been tokenized, or the "
                    "workload bypass the tokenizer but does not set max_tokens_in_prompt by itself."
                )

            # Use max_tokens_in_prompt + max_tokens as the max_model_len. max_tokens_in_prompt
            # is either inferred by materializing tokenized dataset, set by the workload, or
            # set by the engine.
            self.engine_kwargs["max_model_len"] = (
                workload.max_tokens_in_prompt + max_tokens
            )
        else:
            # Sampling params is not given: Embedding workload.
            # In this case, we need to set pooling_params and task.

            if workload.pooling_params is None:
                raise ValueError(
                    "pooling_params is required for vLLM engine for embedding workload."
                )
            constructor_kwargs["params"] = PoolingParams(**workload.pooling_params)
            constructor_kwargs["task"] = "embed"

            # Construct PoolerConfig if override_pooler_config is specified.
            if pooler_config := self.engine_kwargs.get("override_pooler_config", None):
                self.engine_kwargs["override_pooler_config"] = PoolerConfig(
                    **pooler_config
                )

        constructor_kwargs.update(self.engine_kwargs)
        return constructor_kwargs


def init_engine_from_config(
    config: Union[Dict[str, Any], str], override: Optional[Dict[str, Any]] = None
) -> EngineInitializerBase:
    """Initialize an engine initializer from a config file or a config dict.

    Args:
        config: A config file (in YAML) or a config dict. It should include
        the following keys: "engine", backend engine to use; "model",
        model to use; "accelerator_type", the GPU type; "configs",
        the engine specific configurations.
        override: Override values in config["configs"].

    Returns:
        An engine initializer.
    """
    if isinstance(config, str):
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Engine config file {config} not found.")
        with open(config_path, "r") as filep:
            config = yaml.safe_load(filep)

    assert isinstance(config, dict)

    # Override configs
    if override is not None:
        update_dict_recursive(config, override)

    # Ray runtime environments.
    runtime_env: Dict[str, Any] = config.get("runtime_env", {})
    ray_env_vars: Dict[str, Any] = runtime_env.get("env_vars", {})

    # Download model and save to local path in advance, in case
    # too many worker downloads the model in parallel and hit huggingface rate limit.
    assert "model_id" in config and isinstance(config["model_id"], str)
    if ray_env_vars.pop("PREDOWNLOAD_MODEL_FROM_HF", "0") == "1":
        config["model_id"] = download_model_from_hf(
            config["model_id"], "/mnt/cluster_storage"
        )

    # Do not download LoRA adapter here because it is not used in the driver node.
    lora_adapter = None
    if "lora_config" in config:
        lora_adapter = config["lora_config"].get("dynamic_lora_loading_path", None)

    # Sanity check for engine kwargs.
    for key in ("llm_engine", "model_id", "accelerator_type"):
        if key not in config:
            raise KeyError(f"Required {key} not found in config.")
    if "engine_kwargs" not in config:
        config["engine_kwargs"] = {}

    name = config["llm_engine"]
    if name == "vllm":
        return vLLMEngineInitializer(
            model_id=config["model_id"],
            accelerator_type=config["accelerator_type"],
            engine_kwargs=config["engine_kwargs"],
            lora_adapter=lora_adapter,
            ray_env_vars=ray_env_vars,
        )

    raise ValueError(f"Unknown engine: {name}")
