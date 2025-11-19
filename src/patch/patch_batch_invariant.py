import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm.model_executor.layers.layernorm import RMSNorm
from typing import Optional, Union


def patch_rms_norm():
    def _forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.forward_native(x, residual)
    RMSNorm.forward_cuda = _forward_cuda

    

def patch_batch_invariant():
    from bio.batch_invariant_ops import enable_batch_invariant_mode
    enable_batch_invariant_mode()
    os.environ["VLLM_USE_V1"] = "1"
    patch_rms_norm()
    # TODO: patch triton attention
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
    # patch_triton_attention()
    print("Successfully patched vLLM to use custom batch_invariant implementation")

# ============================================================================
# Registration
# ============================================================================

_batch_invariant_backward_mode = False
_batch_invariant_backward_lib = None


def enable_batch_invariant_backward_mode():
    """Enable batch invariant backward mode to support gradients.

    This function adds backward pass support to vLLM's existing batch_invariant
    implementations by registering the backward operations. vLLM handles all the
    forward passes, we just add gradient support.
    """
    global _batch_invariant_backward_mode, _batch_invariant_backward_lib

    if _batch_invariant_backward_mode:
        return

    # Get vLLM's batch_invariant library (already created by init_batch_invariance)
    from bio import batch_invariant_ops as bi_ops

    if (
        not hasattr(bi_ops, "_batch_invariant_LIB")
        or bi_ops._batch_invariant_LIB is None
    ):
        raise RuntimeError(
            "vLLM's batch_invariant mode is not initialized. "
            "enable batch_invariant_mode first."
        )

    # Use vLLM's existing library - don't destroy it!
    _batch_invariant_backward_lib = bi_ops._batch_invariant_LIB
    from torchtitan.experiments.deterministic_vllm_rl.batch_invariant.batch_invariant_backward import matmul_backward_impl, linear_backward_impl
    # Just add the backward operations - everything else is already handled by vLLM
    _batch_invariant_backward_lib.impl(
        "aten::matmul_backward", matmul_backward_impl, "CUDA"
    )
    _batch_invariant_backward_lib.impl(
        "aten::linear_backward", linear_backward_impl, "CUDA"
    )

    _batch_invariant_backward_mode = True


def disable_batch_invariant_backward_mode():
    """Disable batch invariant backward mode."""
    global _batch_invariant_backward_mode, _batch_invariant_backward_lib

    if _batch_invariant_backward_lib is not None:
        _batch_invariant_backward_lib._destroy()

    _batch_invariant_backward_mode = False
    _batch_invariant_backward_lib = None


def is_batch_invariant_backward_mode_enabled():
    """Check if batch invariant backward mode is enabled."""
    return _batch_invariant_backward_mode