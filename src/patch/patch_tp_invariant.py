import torch
from typing import Optional, Union
from torch.nn.parameter import Parameter
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.distributed import split_tensor_along_last_dim



def patch_row_linear():
    def _forward(
            self, input_
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[self.tp_rank].contiguous()

        # Matrix multiply.
        assert self.quant_method is not None
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias

        from tbik.tree_based_matmul import matmul_tp_persistent
        from vllm.distributed.parallel_state import get_tp_group
        from tbik.tree_based_all_reduce import tree_all_reduce_sum

        output_parallel = matmul_tp_persistent(input_parallel, self.weight.t())

        if self.reduce_results and self.tp_size > 1:
            output = tree_all_reduce_sum(output_parallel, device_group=get_tp_group().device_group)
            # output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        if (not self.skip_bias_add) and self.bias is not None:
            output += bias_

        output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
            return output
        return output, output_bias

    RowParallelLinear.forward = _forward

def patch_tp_invariant():
    patch_row_linear()
    print("Successfully patched vLLM to use custom tp_invariant implementation")