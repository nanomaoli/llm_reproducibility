from src.patch import apply_patches
apply_patches()
from src.utils import batch_invariant_is_enabled, tp_invariant_is_enabled
import torch
import random

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

model_path = 'Qwen/Qwen3-1.7B'

temperature = 0.7
logprobs = 10
generate_tokens = 64
seed = 42

def _random_prompt(min_words: int = 1024, max_words: int = 1024 * 2) -> str:
    # Generate more realistic prompts that will actually produce varied tokens
    # Use a mix of common English text patterns

    prompt_templates = [
        # Question-answer style
        "Question: What is the capital of France?\nAnswer: The capital of France is",
        "Q: How does photosynthesis work?\nA: Photosynthesis is the process by which",
        "User: Can you explain quantum mechanics?\nAssistant: Quantum mechanics is",
        # Story/narrative style
        "Once upon a time in a distant galaxy, there lived",
        "The old man walked slowly down the street, remembering",
        "In the year 2157, humanity finally discovered",
        # Technical/code style
        "To implement a binary search tree in Python, first we need to",
        "The algorithm works by iterating through the array and",
        "Here's how to optimize database queries using indexing:",
        # Factual/informative style
        "The Renaissance was a period in European history that",
        "Climate change is caused by several factors including",
        "The human brain contains approximately 86 billion neurons which",
        # Conversational style
        "I've been thinking about getting a new laptop because",
        "Yesterday I went to the store and bought",
        "My favorite thing about summer is definitely",
    ]

    # Pick a random template
    base_prompt = random.choice(prompt_templates)

    if max_words < min_words:
        max_words = min_words
    target_words = random.randint(min_words, max_words)

    if target_words > 50:
        # For longer prompts, repeat context
        padding_text = (
            " This is an interesting topic that deserves more explanation. "
            * (target_words // 50)
        )
        base_prompt = base_prompt + padding_text

    return base_prompt


def vllm_generate(tp_size, batch_size, prompts):
    token_id_vllm = []
    text_vllm = []
    logprob_vllm = []

    sampling_param = SamplingParams(
    temperature=temperature,
    max_tokens=generate_tokens,
    logprobs=logprobs,
    seed=seed,
)
    llm = LLM(
        model_path,
        dtype='bfloat16',
        enforce_eager=True,
        enable_prefix_caching=False,
        max_logprobs=logprobs,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.5,
        logprobs_mode='raw_logprobs',
    )

    for i in range(len(prompts) // batch_size):
        input_prompt = prompts[i * batch_size: min(len(prompts), (i + 1) * batch_size)]
        outputs = llm.generate(input_prompt, sampling_param)

        for output in outputs:
            text = output.outputs[0].text
            token_ids = output.outputs[0].token_ids
            logprob = output.outputs[0].logprobs

            # logprob: list[step] -> dict[token_id -> Logprob]
            converted = [
                [(token_id, lp.logprob) for token_id, lp in step.items()]
                for step in logprob
            ]

            logprob_vllm.append(converted)
            token_id_vllm.append(token_ids)
            text_vllm.append(text)

    del llm
    cleanup_dist_env_and_memory()
    return token_id_vllm, text_vllm, logprob_vllm


def build_prob_tensor(logprob_vllm):
    # logprob_vllm: list[num_samples] of list[steps] of list[(token_id, logprob)]
    logprob_tensor = torch.tensor(
        [
            [
                [t[1] for t in seq[:logprobs]]  # Take the first logprobs candidates from each step
                for seq in sample
            ]
            for sample in logprob_vllm
        ],
        dtype=torch.float32,
    ).cuda()
    prob_tensor = torch.exp(logprob_tensor)
    return prob_tensor


def print_setting_outputs(setting_name, prompts, texts):
    print(f'=== {setting_name} outputs ===')
    for p, t in zip(prompts, texts):
        print(f'Prompt : {p}')
        print(f'Output : {t}')
        print('-' * 40)
    print()


if __name__ == '__main__':
    # Three different settings
    tp_size_1, batch_size_1 = 1, 4
    tp_size_2, batch_size_2 = 4, 4
    tp_size_3, batch_size_3 = 4, 2

    prompts = [_random_prompt(10, 50) for i in range(4)]
    # Setting 1
    token_ids_1, texts_1, logprob_vllm_1 = vllm_generate(tp_size_1, batch_size_1, prompts)
    prob_vllm_tensor_1 = build_prob_tensor(logprob_vllm_1)

    # Setting 2
    token_ids_2, texts_2, logprob_vllm_2 = vllm_generate(tp_size_2, batch_size_2, prompts)
    prob_vllm_tensor_2 = build_prob_tensor(logprob_vllm_2)

    # Setting 3
    token_ids_3, texts_3, logprob_vllm_3 = vllm_generate(tp_size_3, batch_size_3, prompts)
    prob_vllm_tensor_3 = build_prob_tensor(logprob_vllm_3)

    # Print environment modes
    print(f'Batch-Invariant Enabled: {batch_invariant_is_enabled()}')
    print(f'TP-Invariant Enabled   : {tp_invariant_is_enabled()}')
    print()

    # Print the actual outputs of each setting
    print_setting_outputs(
        f'Setting 1 (tp={tp_size_1}, batch_size={batch_size_1})',
        prompts,
        texts_1,
    )
    print_setting_outputs(
        f'Setting 2 (tp={tp_size_2}, batch_size={batch_size_2})',
        prompts,
        texts_2,
    )
    print_setting_outputs(
        f'Setting 3 (tp={tp_size_3}, batch_size={batch_size_3})',
        prompts,
        texts_3,
    )

    # Print the Max Difference Probability between the three settings
    def max_diff(a, b):
        return (a - b).abs().max().item() * 100

    print('-' * 50)
    print('Max Difference Probability (pairwise):')
    print(
        f'  Setting1 (tp={tp_size_1}, bs={batch_size_1}) '
        f'vs Setting2 (tp={tp_size_2}, bs={batch_size_2}): '
        f'{max_diff(prob_vllm_tensor_1, prob_vllm_tensor_2):.6f}%'
    )
    print(
        f'  Setting1 (tp={tp_size_1}, bs={batch_size_1}) '
        f'vs Setting3 (tp={tp_size_3}, bs={batch_size_3}): '
        f'{max_diff(prob_vllm_tensor_1, prob_vllm_tensor_3):.6f}%'
    )
    print(
        f'  Setting2 (tp={tp_size_2}, bs={batch_size_2}) '
        f'vs Setting3 (tp={tp_size_3}, bs={batch_size_3}): '
        f'{max_diff(prob_vllm_tensor_2, prob_vllm_tensor_3):.6f}%'
    )
