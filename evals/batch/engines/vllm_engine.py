"""The vLLM engine."""

import asyncio
import dataclasses
import math
import os
import sys
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import msgspec
import numpy as np
import ray
from packaging import version
from vllm import AsyncEngineArgs, AsyncLLMEngine, PoolingParams, SamplingParams
from vllm.inputs.data import TextPrompt, TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput, RequestOutput

from ..logging import get_logger
from ..utils import (
    async_caller_empty_batch_handler,
    maybe_download_model_from_s3,
    wait_for_gpu_memory_to_clear,
)
from .base import EngineBase

logger = get_logger(__name__)


@dataclass(frozen=True)
class LLMRequest:
    """A request to the LLM wrapper."""

    # Index in the batch.
    idx_in_batch: int
    # The request ID for the LLM engine (unique per replica).
    request_id: int
    # The full prompt string (with chat template applied if any).
    prompt: str
    # The tokenized prompt IDs. If None, then the string prompt will be
    # tokenized by the LLM engine. This is not recommended for performance reasons.
    prompt_token_ids: Optional[List[int]]
    # The sampling or pooling parameters.
    params: Union[SamplingParams, PoolingParams]
    # Custom data to be passed through to the output.
    custom_data: Dict[str, Any]
    # (optional) LoRA adapter.
    lora_request: Optional[LoRARequest] = None


class AsyncLLMWrapper:
    """Wrapper around the vLLM engine to handle async requests.

    Args:
        *args: The positional arguments for the engine.
        max_pending_requests: The maximum number of pending requests in the queue.
        **kwargs: The keyword arguments for the engine.
    """

    def __init__(self, *args, max_pending_requests: int = -1, **kwargs):
        engine_args = AsyncEngineArgs(
            *args,
            **kwargs,
            disable_log_requests=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.max_pending_requests = max_pending_requests

        # Determine the generate function based on vLLM v0 or v1.
        if os.getenv("VLLM_USE_V1", "0") == "1":
            self._generate_async = self.generate_async_v1
        else:
            self._generate_async = self.generate_async_v0

        # FIXME: The asyncio queue crashes in Python 3.9 and Ray 2.37- with the following error:
        # got Future <Future pending> attached to a different loop, because Ray Data
        # creates a new event loop. This can be removed when
        # https://github.com/ray-project/ray/issues/47734 is released.
        if (
            version.parse(ray.__version__) <= version.parse("2.37.0")
            and sys.version_info.minor == 9
        ):
            if self.max_pending_requests > 0:
                logger.warning(
                    "max_pending_requests is disabled due to a known issue with asyncio "
                    "in Python 3.9 with Ray 2.37-"
                )
            self.max_pending_requests = 0

        # vLLM performance gets really bad if there are too many requests in the pending queue.
        # We work around it by introducing another queue that gates how many requests we are
        # sending to vLLM at once.
        # This is not a queue of requests - instead, this queue holds "slots". Each time
        # we add a new request, we take one slot. Each time a request finishes, we add a new
        # slot.
        self.free_queue: asyncio.Queue[bool] = asyncio.Queue()
        if self.max_pending_requests > 0:
            for _ in range(self.max_pending_requests):
                self.free_queue.put_nowait(True)

    async def generate_async(
        self, request: LLMRequest
    ) -> Tuple[LLMRequest, RequestOutput]:
        """Process a single request.

        Args:
            request: The request.

        Returns:
            A tuple of index in batch, request output and bypassed custom fields.
        """
        # If free queue is used, guard the request here until a slot is available.
        if self.max_pending_requests > 0:
            await self.free_queue.get()

        ret = await self._generate_async(request)

        # If free queue is used, release the slot.
        if self.max_pending_requests > 0:
            self.free_queue.put_nowait(True)

        return ret

    async def generate_async_v0(
        self, request: LLMRequest
    ) -> Tuple[LLMRequest, RequestOutput]:
        """Process a single request.

        Args:
            request: The request.

        Returns:
            A tuple of index in batch, request output and bypassed custom fields.
        """
        if request.prompt_token_ids is not None:
            llm_prompt = TokensPrompt(prompt_token_ids=request.prompt_token_ids)
        else:
            assert request.prompt
            llm_prompt = TextPrompt(prompt=request.prompt)

        # Send the request to the LLM engine.
        stream = await self.engine.add_request(
            request_id=str(request.request_id),
            prompt=llm_prompt,
            params=request.params,
            lora_request=request.lora_request,
        )
        # Consume the stream until the request is finished.
        async for request_output in stream:
            if request_output.finished:
                # Bypass the original full prompt.
                request_output.prompt = request.prompt
                return (request, request_output)
        raise RuntimeError("Should not reach here")

    async def generate_async_v1(
        self, request: LLMRequest
    ) -> Tuple[LLMRequest, RequestOutput]:
        """Process a single request.

        Args:
            request: The request.

        Returns:
            A tuple of index in batch, request output and bypassed custom fields.
        """
        # NOTE: vLLM v1 tighly couples tokenizer and detokenizer to the engine,
        # so we should set tokenize=False in .run() to avoid redundant tokenization
        # for better performance (although the impact should be minimal).
        assert request.prompt
        llm_prompt = TextPrompt(prompt=request.prompt)

        # Send the request to the LLM engine.
        stream = self.engine.generate(
            request_id=str(request.request_id),
            prompt=llm_prompt,
            sampling_params=request.params,
            lora_request=request.lora_request,
        )

        # Consume the stream until the request is finished.
        async for request_output in stream:
            if request_output.finished:
                # Bypass the original full prompt.
                request_output.prompt = request.prompt
                return (request, request_output)

        raise RuntimeError("Should not reach here")


class AsyncLLMPredictor(EngineBase):
    """Async LLM predictor.

    Args:
        model: The model name.
        params: The sampling or pooling parameters.
        lora_adapter: The LoRA adapter.
        max_pending_requests: The maximum number of pending requests.
        **kwargs: The keyword arguments for the engine.
    """

    def __init__(
        self,
        model: str,
        params: Union[SamplingParams, PoolingParams],
        lora_adapter: Optional[str] = None,
        max_pending_requests: Optional[int] = None,
        **kwargs,
    ):
        # Sanity check.
        for key in (
            "enable_prefix_caching",
            "enforce_eager",
            "pipeline_parallel_size",
            "tensor_parallel_size",
            "max_num_seqs",
        ):
            assert key in kwargs, f"[InternalError] {key} not found in engine_kwargs."

        # Download model from S3 if needed.
        model = maybe_download_model_from_s3(model)
        # Download LoRA adapter from S3 if needed.
        if lora_adapter is not None:
            lora_adapter = maybe_download_model_from_s3(lora_adapter)

        wait_for_gpu_memory_to_clear(1000 * 2**20)
        self.request_id = 0
        self.enable_prefix_caching = kwargs["enable_prefix_caching"]
        self.params = params
        self.lora_request = (
            LoRARequest("adapter", 1, lora_adapter)
            if lora_adapter is not None
            else None
        )
        if self.lora_request is not None:
            logger.info("LoRA adapter is enabled: %s", lora_adapter)
            # Enforce enable_lora=True in the engine kwargs
            kwargs["enable_lora"] = True

        # Set max_logprobs to the maximum of sampling_logprobs and sampling_prompt_logprobs.
        if isinstance(params, SamplingParams):
            sampling_logprobs = params.logprobs or 0
            sampling_prompt_logprobs = params.prompt_logprobs or 0
            kwargs["max_logprobs"] = max(sampling_logprobs, sampling_prompt_logprobs)

        attn_backend = os.getenv("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
        if attn_backend == "FLASHINFER" and self.enable_prefix_caching:
            if kwargs["kv_cache_dtype"].startswith("fp8"):
                # FlashInfer does not support bfloat16 activations.
                kwargs["dtype"] = "float16"

        self.enforce_eager = kwargs["enforce_eager"]
        pp_size = kwargs["pipeline_parallel_size"]
        self.max_pending_requests = max_pending_requests or math.ceil(
            kwargs["max_num_seqs"] * pp_size * 1.1
        )
        if self.max_pending_requests > 0:
            logger.info("Max pending requests is set to %d", self.max_pending_requests)

        # Create an LLM.
        self.llm = AsyncLLMWrapper(
            model=model,
            disable_log_stats=False,
            max_pending_requests=self.max_pending_requests,
            **kwargs,
        )

    def _prepare_llm_input(self, batch: Dict[str, Any]) -> List[LLMRequest]:
        """Prepare the inputs for LLM inference.

        Args:
            batch: The batch.

        Returns:
            A list of LLMRequest.
        """
        if "prompt" not in batch:
            raise ValueError(
                "Required 'prompt' not found in batch. This may be "
                "due to an unknown internal error if your workload needs "
                "tokenization. If your workload does not need tokenization, "
                "please make sure 'prompt' exists in the dataset."
            )
        prompt = batch.pop("prompt").tolist()

        if "tokenized_prompt" in batch:
            tokenized_prompt = batch.pop("tokenized_prompt").tolist()
        else:
            tokenized_prompt = [None] * len(prompt)

        # If sampling_params is provided in the batch, override the default.
        if "sampling_params" in batch:
            sampling_params_dict = batch.pop("sampling_params").tolist()
            params = [SamplingParams(**s) for s in sampling_params_dict]
        else:
            params = [self.params] * len(prompt)

        # Rest fields are custom data.
        keys, values = list(batch.keys()), zip(*batch.values())
        custom_data = [dict(zip(keys, v)) for v in values]

        # Organize data to be LLM requests.
        requests = []
        for idx, (p, pt, sp, cd) in enumerate(
            zip(prompt, tokenized_prompt, params, custom_data)
        ):
            requests.append(
                LLMRequest(
                    idx_in_batch=idx,
                    request_id=self.request_id,
                    prompt=p,
                    prompt_token_ids=pt,
                    params=sp,
                    custom_data=cd,
                    lora_request=self.lora_request,
                )
            )
            self.request_id += 1
        return requests

    def _parse_llm_output(
        self, output: Union[RequestOutput, PoolingRequestOutput]
    ) -> Dict[str, Any]:
        """Parse the LLM output.

        Args:
            output: The LLM output.

        Returns:
            The parsed output.
        """
        # Parse the common fields.
        output_data = {
            "prompt": [output.prompt],
            "prompt_token_ids": [output.prompt_token_ids],
            "num_input_tokens": [len(output.prompt_token_ids)],
            "request_id": [output.request_id],
        }

        if isinstance(output, RequestOutput):
            metrics = {}
            if output.metrics is not None:
                metrics = {
                    k: [v] for k, v in dataclasses.asdict(output.metrics).items()
                }
            generated_tokens = [
                output.outputs[i].token_ids for i in range(len(output.outputs))
            ]
            num_generated_tokens = [
                len(output.outputs[i].token_ids) for i in range(len(output.outputs))
            ]
            output_data.update(
                {
                    "generated_tokens": (
                        [generated_tokens]
                        if len(generated_tokens) > 1
                        else generated_tokens
                    ),
                    "num_generated_tokens": (
                        [num_generated_tokens]
                        if len(num_generated_tokens) > 1
                        else num_generated_tokens
                    ),
                    **metrics,
                }
            )
        elif isinstance(output, PoolingRequestOutput):
            output_data.update(
                {
                    "embeddings": [output.outputs.data.cpu()],
                }
            )
        else:
            raise ValueError(f"Unknown output type: {type(output)}")

        return output_data

    async def call_async(
        self, batch: Dict[str, np.ndarray]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Call the LLM asynchronously to process a batch.

        Args:
            batch: The batch.

        Yields:
            The output.
        """
        batch_uuid = uuid.uuid4()
        t = time.perf_counter()

        requests = self._prepare_llm_input(batch)
        tasks = [
            asyncio.create_task(self.llm.generate_async(request))
            for request in requests
        ]

        time_taken = -1.0
        for resp in asyncio.as_completed(tasks):
            request, output = await resp
            time_taken = time.perf_counter() - t
            index_in_batch = request.idx_in_batch
            param_dict = msgspec.structs.asdict(request.params)
            # Convert RequestOutputKind (Enum) to integer value.
            if "output_kind" in param_dict and isinstance(
                param_dict["output_kind"], Enum
            ):
                param_dict["output_kind"] = param_dict["output_kind"].value
            custom_data = request.custom_data
            custom_data = {k: [v] for k, v in custom_data.items()}

            yield {
                **self._parse_llm_output(output),
                "batch_uuid": [batch_uuid.hex],
                "time_taken_llm": [time_taken],
                "index_in_batch": [index_in_batch],
                "params": [param_dict],
                **custom_data,
            }
        logger.info(
            "[vLLM] Elapsed time for batch %s with size %d: %s",
            batch_uuid.hex,
            len(requests),
            time_taken,
        )

    @async_caller_empty_batch_handler
    async def __call__(
        self, batch: Dict[str, np.ndarray]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Call the LLM asynchronously to process a batch.

        Args:
            batch: The batch.

        Yields:
            The output.
        """
        async for x in self.call_async(batch):
            yield x
