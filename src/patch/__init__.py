from .patch_batch_invariant import patch_batch_invariant
from .patch_tp_invariant import patch_tp_invariant
from .patch_training_inference_align import patch_training_inference_align

__all__ = ["apply_patches"]

def apply_patches():
    from src.utils import batch_invariant_is_enabled, tp_invariant_is_enabled, compatible_mode_is_enabled
    if not batch_invariant_is_enabled() and not tp_invariant_is_enabled() and not compatible_mode_is_enabled():
        print("No patches applied to vLLM\n")
        return
    if compatible_mode_is_enabled():
        patch_batch_invariant()
        patch_tp_invariant()
        patch_training_inference_align()
        print("Inference Engine(vLLM) and Training Engine(TorchTitan) are aligned, applying both batch invariance and tp invariance patches\n")
    elif not compatible_mode_is_enabled() and batch_invariant_is_enabled() and not tp_invariant_is_enabled():
        patch_batch_invariant()
        print("Applying batch invariance patches to vLLM\n")
    elif not compatible_mode_is_enabled() and batch_invariant_is_enabled() and tp_invariant_is_enabled():
        patch_batch_invariant()
        patch_tp_invariant()
        print("Applying both batch invariance and tp invariance patches to vLLM\n")
    else:
        raise ValueError("Invalid combination of batch invariance, tp invariance, and compatible mode")
    print("Successfully applied patches\n")
    return