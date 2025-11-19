from typing import Any, Dict, List, Optional, Tuple
import torch
from typing import Optional, Tuple
from vllm.model_executor.layers.rotary_embedding import get_rope, RotaryEmbedding
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
from vllm.attention import get_attn_backend
import torch.nn.functional as F
from vllm.utils import get_dtype_size
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.worker.cache_engine import CacheEngine
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType,
                        get_dtype_size, is_pin_memory_available)
from vllm.distributed import get_pp_group
from vllm.model_executor.models.qwen2 import Qwen2Model, LogitsProcessor, get_sampler, ParallelLMHead, PPMissingLayer, maybe_prefix
from vllm.model_executor.models.llama import LlamaModel, LlamaForCausalLM, LlamaAttention

from vllm.attention import Attention
# from vllm.v1.worker.gpu_model_runner
import sys
import pdb
# from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
# STR_DTYPE_TO_TORCH_DTYPE["float32"] = torch.float32
# from vllm.config import CacheConfig


class ForkedPdb(pdb.Pdb):
    """
    PDB Subclass for debugging multi-processed code
    Suggested in: https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
            
def convert_linear_weights_to_fp16(model: torch.nn.Module):
    """Convert weights of linear layers to fp16 for storage."""
    for name, module in model.named_modules():
        if 'proj' in name:
            module.weight.data = module.weight.data.to(torch.float16)
            if module.bias is not None:
                module.bias.data = module.bias.data.to(torch.float16)

def convert_linear_weights_to_bfloat16(model: torch.nn.Module):
    """Convert weights of linear layers to bfloat16 for storage."""
    for name, module in model.named_modules():
        if 'proj' in name:
            module.weight.data = module.weight.data.to(torch.bfloat16)
            if module.bias is not None:
                module.bias.data = module.bias.data.to(torch.bfloat16)

def our_attn_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    # Input is already in fp32 from previous layer
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output  # Keep in fp32
    
def our_fp32_rope_forward_cuda(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from vllm import _custom_ops as ops
    # Everything is already in fp32, no need for conversion
    if self.cos_sin_cache.device != query.device:
        self.cos_sin_cache = self.cos_sin_cache.to(query.device)

    if offsets is not None:
        ops.batched_rotary_embedding(positions, query, key, self.head_size,
                                   self.cos_sin_cache,
                                   self.is_neox_style, self.rotary_dim,
                                   offsets)
    else:
        ops.rotary_embedding(positions, query, key, self.head_size,
                           self.cos_sin_cache, self.is_neox_style)
    return query, key
    

def our_linear_apply(self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    # x is already in fp32
    assert x.dtype == torch.float32
    # Upcast weights to fp32 for computation
    weight = layer.weight.to(torch.float32)
    if bias is not None:
        bias = bias.to(torch.float32)
    return F.linear(x, weight, bias)  # Result stays in fp32

def patch_cache_engine():
    original_init = CacheEngine.__init__
    def custom_cache_engine_init(
        self,
        cache_config,
        model_config,
        parallel_config,
        device_config,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config

        self.head_size = model_config.get_head_size()
        # Models like Jamba, have mixed typed layers, E.g Mamba
        self.num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        if self.num_gpu_blocks:
            self.num_gpu_blocks //= parallel_config.pipeline_parallel_size
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        if self.num_cpu_blocks:
            self.num_cpu_blocks //= parallel_config.pipeline_parallel_size

        self.dtype = torch.float32  # Force fp32 for cache

        # Get attention backend.
        self.attn_backend = get_attn_backend(self.head_size,
                                             model_config.dtype,
                                             cache_config.cache_dtype,
                                             self.block_size,
                                             model_config.is_attention_free,
                                             use_mla=model_config.use_mla)

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(
            self.num_gpu_blocks, self.device_config.device_type)
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

    @staticmethod
    def our_get_cache_block_size(
        cache_config,
        model_config,
        parallel_config,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)

        dtype = torch.float32  # Force fp32 for cache
        key_cache_entry = num_heads * head_size

        # For MLA there is no value cache, since the latent vector
        # is joint keys and values.
        value_cache_entry = key_cache_entry if not model_config.use_mla else 0
        total = num_attention_layers * cache_config.block_size * \
            (key_cache_entry + value_cache_entry)

        dtype_size = get_dtype_size(dtype)
        return dtype_size * total

    CacheEngine.__init__ = custom_cache_engine_init
    CacheEngine.get_cache_block_size = our_get_cache_block_size


def patch_qwen2_vllm():
    # from vllm.platforms import _Backend
    # from vllm.attention.selector import global_force_attn_backend
    # global_force_attn_backend(_Backend.XFORMERS)
    import os
    os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS" # FLASHINFER
    from vllm.model_executor.models.qwen2 import Qwen2Attention, Qwen2ForCausalLM
    
    patch_cache_engine()
    
    def new_qwen2_lm_init(self, *, vllm_config, prefix: str = ""):
        torch.nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config
        
        self.model = Qwen2Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              quant_config=quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()

        # Convert linear weights to fp16 for storage
        convert_linear_weights_to_bfloat16(self.model)
        if not isinstance(self.lm_head, PPMissingLayer):
            convert_linear_weights_to_bfloat16(self.lm_head)

        self.logits_processor = LogitsProcessor(config.vocab_size, scale=1.2)
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    Qwen2ForCausalLM.__init__ = new_qwen2_lm_init

    # Store the original __init__
    original_init = Qwen2Attention.__init__
    def new_qwen2_init(self, *args, **kwargs):
        # Call the original init first
        original_init(self, *args, **kwargs)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=kwargs['max_position'],
            base=self.rope_theta,
            rope_scaling=kwargs['rope_scaling'],
            dtype=torch.float32  # RoPE computation in fp32
        )
    
    Qwen2Attention.__init__ = new_qwen2_init
    # Replace the apply method
    UnquantizedLinearMethod.apply = our_linear_apply
    UnquantizedEmbeddingMethod.apply = our_linear_apply
    RotaryEmbedding.forward_cuda = our_fp32_rope_forward_cuda
    Qwen2Attention.forward = our_attn_forward
    print("Patched vLLM for Qwen2: Model loaded in fp32, linear weights stored in bfloat16, all computations in fp32")


def patch_llama_vllm():
    # from vllm.platforms import _Backend
    # from vllm.attention.selector import global_force_attn_backend
    # global_force_attn_backend(_Backend.XFORMERS)
    import os
    os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS" # FLASHINFER
    from vllm.model_executor.models.llama import LlamaAttention, LlamaForCausalLM
    
    patch_cache_engine()
    
    def new_llama_lm_init(self, *, vllm_config, prefix: str = ""):
        torch.nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config
        
        self.model = LlamaModel(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              quant_config=quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()

        # Convert linear weights to fp16 for storage
        convert_linear_weights_to_bfloat16(self.model)
        if not isinstance(self.lm_head, PPMissingLayer):
            convert_linear_weights_to_bfloat16(self.lm_head)

        self.logits_processor = LogitsProcessor(config.vocab_size, scale=1.2)
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    LlamaForCausalLM.__init__ = new_llama_lm_init

    # Store the original __init__
    original_init = LlamaAttention.__init__
    def new_llama_init(self, *args, **kwargs):
        # Call the original init first
        original_init(self, *args, **kwargs)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=kwargs['max_position'],
            base=self.rope_theta,
            rope_scaling=kwargs['rope_scaling'],
            dtype=torch.float32  # RoPE computation in fp32
        )
    
    LlamaAttention.__init__ = new_llama_init
    # Replace the apply method
    UnquantizedLinearMethod.apply = our_linear_apply
    UnquantizedEmbeddingMethod.apply = our_linear_apply
    RotaryEmbedding.forward_cuda = our_fp32_rope_forward_cuda
    LlamaAttention.forward = our_attn_forward
    print("Patched vLLM for Llama: Model loaded in fp32, linear weights stored in bfloat16, all computations in fp32")
