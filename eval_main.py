import vllm
import torch
import logging
import datasets
from vllm import SamplingParams
import os
import argparse
import json
import glob
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from tqdm import tqdm
from evals.tasks import TASK_HANDLER_MAP, TASK_NAMES_TO_YAML, TaskConfig, TaskHandler
from evals.util.results import SummaryResults, save_summary

from vllm.entrypoints.chat_utils import (apply_hf_chat_template,
                                         parse_chat_messages,)
from vllm.utils import is_list_of
from vllm.inputs import TextPrompt, TokensPrompt
from prompt_util.prompt_template import make_conversation_from_contents
from evals.tasks import TASK_HANDLER_MAP, TASK_NAMES_TO_YAML, TaskConfig

logger = logging.getLogger(__name__)

# Add argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Run model inference with configurable parameters')
    parser.add_argument('--model', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
                      help='Model name or path')
    parser.add_argument('--task', type=str, default='math500',
                      help='Task name')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                      help='Data type for model (e.g., bfloat16, float16)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for inference')
    parser.add_argument('--max_tokens', type=int, default=32768,
                      help='Maximum number of tokens to generate')
    parser.add_argument('--exp_name', type=str, default='baseline',
                      help='Experiment name')
    return parser.parse_args()
   


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Determine the starting point based on existing .pt files
def get_resume_point(output_path, batch_size):
    # Find all .pt files in the output directory
    pt_files = glob.glob(f'{output_path}/problem_*_token_ids_*.pt')
    if not pt_files:
        return 0  # No files exist, start from the beginning

    # Extract global_idx from filenames
    global_indices = []
    for pt_file in pt_files:
        # Filename format: problem_<global_idx>_token_ids_*.pt
        parts = os.path.basename(pt_file).split('_')
        try:
            global_idx = int(parts[1])  # Extract the global_idx
            global_indices.append(global_idx)
        except (IndexError, ValueError):
            continue

    if not global_indices:
        return 0  # No valid indices found, start from the beginning

    # Find the largest global_idx and calculate the starting batch
    max_global_idx = max(global_indices)
    resume_point = ((max_global_idx + 1) // batch_size) * batch_size
    print(f"Resuming from batch starting at index {resume_point} (max_global_idx={max_global_idx})")
    return resume_point

def score_responses(
    handler: TaskHandler,
    list_of_results: List[Dict[str, Any]],
    eval_data: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, List[int]], int]:

    if not list_of_results:
        return 0.0, {}, 0

    total_correct = 0
    total_finish = 0
    id_to_scores = {}

    for result in tqdm(list_of_results, desc="Scoring responses"):
        # Get content from the result
        model_response = result['model_answer']
        problem_id = result['problem_id']
        problem = eval_data[problem_id]
        
        new_response_entry = handler.update_results(
            problem=problem,
            response=model_response,
        )
        
        if problem_id not in id_to_scores:
            id_to_scores[problem_id] = [0]
        id_to_scores[problem_id][0] = new_response_entry["correctness"]
        
        total_correct += new_response_entry["correctness"]
        total_finish += 1

    accuracy = round(total_correct / total_finish, 4) if total_finish else 0
    return accuracy, id_to_scores, total_finish

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    # Create outputs directory if it doesn't exist
    output_path = f'./outputs/vllm_main/{args.exp_name}/{args.model}'
    os.makedirs(output_path, exist_ok=True)

    task_config = TaskConfig.from_yaml(TASK_NAMES_TO_YAML[args.task])
    handler_name = task_config.handler
    handler_cls = TASK_HANDLER_MAP[handler_name]
    handler = handler_cls(task_config)
    eval_data = handler.load_and_filter_dataset(0, -1) # start from 0, load all
    remaining_data = handler.process_remaining_data(eval_data, {})
    conversations = handler.make_conversations(
        remaining_data,
        None, # str(model_config.system_prompt),
        None, # model_config.user_template,
        None, # model_config.assistant_prefill,
    )
    total_samples = len(conversations)
    print(f"Total samples in the dataset: {total_samples}")

    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs for tensor parallelism")
    
    model = vllm.LLM(model=args.model, 
                    tensor_parallel_size=num_gpus,
                    # max_model_len=length_used,
                    dtype=args.dtype,
                    enforce_eager=True)
    # Configure sampling parameters to return logits
    sampling_params = SamplingParams(temperature=0.0, logprobs=5, max_tokens=args.max_tokens, seed=args.seed)

    # Process in batches
    qa_pairs = []
    jsonl_path = f'{output_path}/qa_pairs_{args.dtype}_bs_{args.batch_size}.jsonl'
    
    start_point = get_resume_point(output_path, args.batch_size)
    
    for batch_start in range(start_point, total_samples, args.batch_size):
        batch_end = min(batch_start + args.batch_size, total_samples)
        current_batch = conversations[batch_start:batch_end]
        print(f"Processing batch {batch_start//args.batch_size + 1}/{(total_samples + args.batch_size - 1)//args.batch_size}")
        
        tokenizer = model.get_tokenizer()
        model_config = model.llm_engine.get_model_config()
        prompts = []

        for msgs in current_batch:
            # NOTE: _parse_chat_message_content_parts() currently doesn't
            # handle mm_processor_kwargs, since there is no implementation in
            # the chat message parsing for it.
            conversation, mm_data = parse_chat_messages(
                msgs,
                model_config,
                tokenizer,
                content_format='string',
            )

            prompt_data = apply_hf_chat_template(
                tokenizer,
                conversation=conversation,
                chat_template=None,
                add_generation_prompt=True,
                continue_final_message=False,
                tools=None,
            )

            if is_list_of(prompt_data, int):
                prompt = TokensPrompt(prompt_token_ids=prompt_data)
            else:
                prompt = TextPrompt(prompt=prompt_data)

            if mm_data is not None:
                prompt["multi_modal_data"] = mm_data

            prompts.append(prompt)
        
        # Generate with logits for current batch
        response = model.generate(prompts, sampling_params=sampling_params)
        # Extract output text and logits for each sample in the batch
        qa_pairs = []
        for idx, output in enumerate(response):
            global_idx = batch_start + idx
            generated_text = output.outputs[0].text
            token_logprobs = output.outputs[0].logprobs
            # Create tensors from token_logprobs
            num_tokens = len(token_logprobs)
            token_ids = torch.zeros((num_tokens, 5), dtype=torch.long)
            logprobs = torch.zeros((num_tokens, 5), dtype=torch.float32)
            # Save QA pair to JSONL file
            qa_pair = {
                "problem_id": global_idx,
                "question": current_batch[idx],
                "model_answer": generated_text,
            }

            for i, logprobs_dict in enumerate(token_logprobs):
                # Extract token IDs in order of rank
                sorted_items = sorted(logprobs_dict.items(), key=lambda x: x[1].rank)
                for j, (token_id, L) in enumerate(sorted_items):
                    token_ids[i, j] = token_id
                    logprobs[i, j] = L.logprob
            
            torch.save(token_ids, f'{output_path}/problem_{global_idx}_{args.task}_token_ids_bs_{args.batch_size}_{args.dtype}_max_tokens_{args.max_tokens}.pt')
            torch.save(logprobs, f'{output_path}/problem_{global_idx}_{args.task}_logprobs_bs_{args.batch_size}_{args.dtype}_max_tokens_{args.max_tokens}.pt')
            print(f"Saved tensors for problem {global_idx}")
            
            qa_pairs.append(qa_pair)
    
        with open(jsonl_path, 'a') as f:
            for qa_pair in qa_pairs:
                f.write(json.dumps(qa_pair) + '\n')
        print(f"Saved QA pairs to for batch {batch_start//args.batch_size + 1}")
    
    responses_path = Path(jsonl_path)

    if responses_path.stat().st_size == 0:
        raise ValueError(f"Response file is empty: {responses_path}")
        
    print(f"Valid response file: {responses_path}")
    
    # Read the .jsonl file line by line and parse each line as a JSON object
    with open(responses_path, "r") as f:
        list_of_results = [json.loads(line) for line in f]
    
    # Check if the response file is a list of dictionaries
    if not all(isinstance(result, dict) for result in list_of_results):
        raise ValueError(f"Response file does not contain valid dictionaries on each line: {responses_path}")
    
    # Check if the response file is a list of dictionaries
    if not isinstance(list_of_results, list):
        raise ValueError(f"Response file is not a list of dictionaries: {responses_path}")
    
    # Obtain the correct task handler
    task = args.task
    if task not in TASK_NAMES_TO_YAML:
        raise ValueError(
            f"Task {task} not found. Should be one of {TASK_NAMES_TO_YAML.keys()}"
        )
    task_config = TaskConfig.from_yaml(TASK_NAMES_TO_YAML[task])
    handler_name = task_config.handler
    handler_cls = TASK_HANDLER_MAP[handler_name]
    handler = handler_cls(task_config)
    
    raw_dataset = handler.load_and_filter_dataset(0, -1) # start from 0, load all
    eval_data = [
        row.to_dict()
        for _, row in raw_dataset.iterrows()
    ]
    
    accuracy, id_to_scores, total_finish = score_responses(handler, list_of_results, eval_data)
    logger.info(f"Accuracy: {accuracy}")
    
    num_responses_total = len(id_to_scores)

    summary_data = SummaryResults(
        accuracy=accuracy,
    )
    
    # Create outputs directory if it doesn't exist
    acc_path = f'./scoring_results/greedy'
    os.makedirs(acc_path, exist_ok=True)
    
    sanitized_model_name = args.model.replace("/", "_")
    summary_file = Path(acc_path) / f"{sanitized_model_name}_{args.exp_name}_summary.jsonl"
    save_summary(summary_file, summary_data)
    logger.info(f"Summary saved to {summary_file}")
