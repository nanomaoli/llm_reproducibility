# ⚙️ Monkey Patching Solutions for Deterministic LLM Generation

*TL;DR: A toolkit that enables deterministic LLM inference and eliminates the training–inference mismatch in reinforcement learning.*


**Deterministic Inference across Tensor Parallel Sizes That Eliminates Training–Inference Mismatch** [[Paper]()] [[Code]()]

**Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference** (NeurIPS 2025, **Oral**) [[Paper](https://arxiv.org/abs/2506.09501)] [[HF](https://huggingface.co/papers/2506.09501)] [[Code](https://github.com/nanomaoli/llm_reproducibility/tree/main/evaluation)]

## News
- [2025.11.18]: 🗣️ A new paper has been released on [arxiv](). In this paper, we proposed TBIK(Tree Based Invariant Kernels), which enables deterministic inference across TP sizes.
This kernel also fundamentally solves the training–inference mismatch problem in reinforcement learning when they are using different parallelization stragey.
- [2025.09.25]: 🎉🎉🎉 Our paper has been selected for Oral Presentation for Neurips 2025. See you in SD! 
- [2025.06.18]: Our paper has been released on [arxiv](https://arxiv.org/abs/2506.09501). Feel free to ⭐UPVOTE in [huggingface](https://huggingface.co/papers/2506.09501)

## Overview

In our [LLM evaluation reproducibility report](https://arxiv.org/pdf/2506.09501), we found that chaning Batch size and GPU counts can impact the reasoning trace a lot This is due to the differences in float-point arthmetic order across configurations. Thinking Machines Lab introduced [batch-invariant operators](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), which ensure deterministic outputs across different batch sizes. However, nondeterminism remains unresolved when changing TP sizes, a common scenario in RL that training and rollout engines employ different parallelization strategy.

<p align="center">
  <img src="figures/kernel.png" width="400"/>
</p>

<p align="center">
  <i>Illustration of TP-invariant matrix multiplication.</i>
</p>

We propose TBIK, a TP-invariant matmul method that achieves determinism by strictly controlling the reduction order in matrix multiplications. By replacing TP-size-sensitive(Row-Split) layers such as o_proj and down_proj with our TP-invariant counterparts, and by employing a tree-structured cross-GPU all-reduce, we achieve fully deterministic LLM inference across different TP sizes.

Furthermore, we align training engine FSDP (TP=1) with the vLLM rollout engine under tensor parallelism (TP>1), enabling bitwise-identical true on-policy reinforcement learning.

<p align="center">
  <img src="figures/overview.png" width="700"/>
</p>

<p align="center">
  <i>Illustration of global reductions within and across GPUs for Row-Split layers in Transformer Models.</i>
</p>


## Setup
1. Install vllm
```bash
conda create -n tbik python=3.12 -y
conda activate tbik
pip install vllm==0.11.0
```
<!-- 2. Install dependencies
```bash
pip install datasets latex2sympy2 word2number immutabledict nltk langdetect
``` -->

2. Install Torchtitan and Flash Attention(This is optional unless you want to try `simple_rl.py`)
```bash
# Flash Attention
pip install flash-attn --no-build-isolation
# Torchtitan
git clone https://github.com/xh-ding/torchtitan.git
cd torchtitan
pip install -e .
cd ..
```

3. Download this repository
```bash
git clone https://github.com/nanomaoli/llm_reproducibility
cd llm_reproducibility
```

## How to use TBIK?
Unlock deterministic vLLM inference and true policy reinforcement learning with just a single, lightweight `apply_patches()` function! It automatically applies the appropriate patches based on the environment variables.

### Deterministic Matmul
Matmul operations have different reduction order when splitting over different numbers of GPUs. With our tree-based matmul, the result of matrix multiplication remains invariant regardless of the number of splits. You can switch between the standard matmul and the tree-based matmul by setting `TP_INVARIANT_MATMUL=0/1`.
```bash
TP_INVARIANT_MATMUL=1 python simple_matmul.py
```

### Deterministic LLM inference
In vLLM’s model implementation, the `o_proj` layer in the attention module and the `down_proj` layer in the FFN use 'row-split' linear layers. By replacing these row-split linear layers with our TP-invariant counterparts and co-designing both the intra-kernel reduction order and the inter-GPU reduction order(and also batch invariant operations), vLLM can achieve deterministic inference across different runtime settings.
```bash
VLLM_BATCH_INVARIANT=1 VLLM_TP_INVARIANT=1 python simple_inference.py
```

### Bitwise consistent on-policy RL
Following [spirl](https://github.com/bwasti/spirl), we use vLLM for inference and TorchTitan for training in our demo code to make the workflow easier to try for users. In practical RL training pipelines, the training engine typically runs with FSDP (TP = 1), while the inference engine runs with TP > 1. This mismatch in numerical precision across the two engines is the fundamental cause of training instability in RL. By introducing TBIK and aligning the operators used in both the training and inference engines, we address this issue at its root, making true on-policy RL feasible.
```bash
[CUDA_VISIBLE_DEVICES] VLLM_BATCH_INVARIANT=1 VLLM_TP_INVARIANT=1 ALIGN_TRAIN_INFERENCE=1 python simple_rl.py
```

## Evaluating Reproducibility of Reasoning
To run the evaluation on the full dataset, please refer to [./evaluation](https://github.com/nanomaoli/llm_reproducibility/tree/main/evaluation)

## Contributing
The code was currently tested with ✅Qwen3-1.7B, should work with other Qwen3 models with the same architecture.

We welcome contributions from the research community to improve TBIK. If you have any idea or would like to report a bug, please open an issue or submit a pull request.


## Citation

If you find our work helpful, please kindly cite our paper.

```bibtex

@misc{yuan2025fp32deathchallengessolutions,
      title={Give Me FP32 or Give Me Death? Challenges and Solutions for Reproducible Reasoning}, 
      author={Jiayi Yuan and Hao Li and Xinheng Ding and Wenya Xie and Yu-Jhe Li and Wentian Zhao and Kun Wan and Jing Shi and Xia Hu and Zirui Liu},
      year={2025},
      eprint={2506.09501},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.09501}, 
}
```

## Acknowledgment
Our implementation of `simple_rl.py` and `torchtitan` is adapted from [spirl](https://github.com/bwasti/spirl) repository.
