import vllm
import torch
import datasets
from vllm import SamplingParams
import os
import argparse
import json
import glob
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from tqdm import tqdm
import statistics
from evals.tasks import TASK_HANDLER_MAP, TASK_NAMES_TO_YAML, TaskConfig, TaskHandler
from evals.util.results import SummaryResults, save_summary
from evals.util.metrics import pass_at_k

from vllm.entrypoints.chat_utils import (apply_hf_chat_template,
                                         parse_chat_messages,)
from vllm.utils import is_list_of
from vllm.inputs import TextPrompt, TokensPrompt
from prompt_util.prompt_template import make_conversation_from_contents
from evals.tasks import TASK_HANDLER_MAP, TASK_NAMES_TO_YAML, TaskConfig

logger = logging.getLogger(__name__)

TASK_MAX_IDX = {
    'aime24': 29,
    'math500': 499,
    'livecodebench_easy': 181,
    'livecodebench_medium': 205,
    'livecodebench_hard': 122,
}

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
    parser.add_argument('--passk', type=int, default=1,
                      help='number of output sequences to generate for each input')
    parser.add_argument('--exp_name', type=str, default='baseline',
                      help='Experiment name')
    return parser.parse_args()
   


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Determine the starting point based on existing qa_pairs file
def get_resume_point(output_path, batch_size, task, dtype, passk):
    qa_file_path = os.path.join(output_path, f"qa_pairs_{dtype}_bs_{batch_size}_pass{passk}.jsonl")  # Modify if filename differs

    if not os.path.exists(qa_file_path):
        return 0  # Start from the beginning

    try:
        with open(qa_file_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                return 0  # File is empty
            last_line = lines[-1].strip()
            if not last_line:
                return 0
            last_entry = json.loads(last_line)
            max_global_idx = last_entry["problem_id"]
    except Exception as e:
        print(f"Error reading qa_pairs file: {e}")
        return 0

    resume_point = ((max_global_idx + 1) // batch_size) * batch_size

    if task in TASK_MAX_IDX and max_global_idx == TASK_MAX_IDX[task]:
        exit(0)  # All samples have been processed

    print(f"Resuming from batch starting at index {resume_point} (max_global_idx={max_global_idx})")
    return resume_point

def score_responses(
    handler: TaskHandler,
    list_of_results: List[Dict[str, Any]],
    eval_data: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, List[int]], int]:

    if not list_of_results:
        return 0.0, {}, 0
    id_to_results: Dict[int, Dict[str, Any]] = {}

    total_correct = 0
    total_finish = 0
    id_to_scores = {}

    for result in tqdm(list_of_results, desc="Scoring responses"):
        # Get content from the result
        model_responses = result['model_answers']
        problem_id = result['problem_id']
        
        problem = eval_data[problem_id]
        
        if not isinstance(model_responses, list):
            model_responses = [model_responses] # in case it's a single string

        # build a model_responses, which is a list of dicts, aligned with SkyThought
        id_to_results[problem_id] = {
            "responses": [{"content": response, "correctness": None, "reason": None} for response in model_responses],
        }
        
        scores = []
        for i, response_obj in enumerate(id_to_results[problem_id]["responses"]):
            content = response_obj["content"]
            new_response_entry = handler.update_results(
                problem=problem,
                response=content,
            )
            response_obj["correctness"] = new_response_entry["correctness"]
            response_obj["reason"] = new_response_entry["reason"]
        
            if problem_id not in id_to_scores:
                id_to_scores[problem_id] = [0 for _ in range(args.passk)]
            id_to_scores[problem_id][i] = new_response_entry["correctness"]
        
            total_correct += new_response_entry["correctness"]
            total_finish += 1

    accuracy = round(total_correct / total_finish, 4) if total_finish else 0
    return accuracy, id_to_scores, total_finish

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    # Create outputs directory if it doesn't exist
    output_path = f'./outputs/vllm_passk/{args.exp_name}/{args.model}'
    os.makedirs(output_path, exist_ok=True)
    
    start_point = get_resume_point(output_path, args.batch_size, args.task, args.dtype, args.passk)

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
    sampling_params = SamplingParams(n=args.passk, temperature=0.7, top_p=0.95, logprobs=5, max_tokens=args.max_tokens, seed=args.seed)

    # Process in batches
    qa_pairs = []
    jsonl_path = f'{output_path}/qa_pairs_{args.dtype}_bs_{args.batch_size}_pass{args.passk}.jsonl'
    
   
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
            all_generated_text = []
            
            for ans_id, ans in enumerate(output.outputs):
                # ans_id is the index of the answer in all answers
                # output.outputs[ans_id] == ans
                all_generated_text.append(ans.text)
            # save all answer samples
            qa_pair = {
                "problem_id": global_idx,
                "question": current_batch[idx],
                "model_answers": all_generated_text, # this should be a list of answers
            }
            
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
    pass_at_k_metrics = None
    pass_at_k_metrics = pass_at_k(args.passk, id_to_scores)
    
    num_responses_total = len(id_to_scores)

    summary_data = SummaryResults(
        accuracy=accuracy,
        pass_at_k=pass_at_k_metrics,
    )
    
    # Create outputs directory if it doesn't exist
    acc_path = f'./scoring_results/random_passk'
    os.makedirs(acc_path, exist_ok=True)
    
    sanitized_model_name = args.model.replace("/", "_")
    summary_file = Path(acc_path) / f"{sanitized_model_name}_{args.exp_name}_summary.jsonl"
    save_summary(summary_file, summary_data)
    logger.info(f"Summary saved to {summary_file}")
