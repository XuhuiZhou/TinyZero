"""
Preprocess dataset for TOMI test - Theory of Mind reasoning task
"""

import os
import json
from datasets import Dataset
from typing import List
from verl.utils.hdfs_io import copy, makedirs
import argparse


def load_tomi_data(file_path: str) -> List[dict]:
    """Load and parse TOMI test data from JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def make_prefix(example, template_type):
    """Create prompt prefix based on template type."""
    question = example['question']
    story = example['story']
    
    task_assumptions = """Assume that the characters can perceive every scene in their location but not scenes occurring elsewhere. If something is being moved, that means it is not in its original location anymore. You should majorly focus on where the object has been moved to, and answer the question with the most detailed position possible e.g., the object is in A and A is in B, then you should answer 'A'."""
    
    if template_type == 'base':
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: {task_assumptions} Here is the story: {story} And here is the question: {question}
Please show your reasoning in <think> </think> tags and provide shortest possible final answer in <answer> </answer> tags.
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n{task_assumptions}\nStory: {story}\nQuestion: {question}\nPlease show your reasoning in <think> </think> tags and provide shortest possible final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix

def process_tomi_data(example, idx, split, template_type='base', data_source='tomi'):
    """Process TOMI test data into standard format."""
    question = make_prefix(example, template_type)
    solution = {
        "answer": example['answer'],
        "story": example['story']
    }
    
    return {
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": question,
        }],
        "ability": "reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": solution
        },
        "extra_info": {
            'split': split,
            'index': idx,
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', type=str, default="./data/tomi_data", 
                       help='Local directory to save processed data')
    parser.add_argument('--hdfs_dir', type=str, default=None,
                       help='HDFS directory to copy data to')
    parser.add_argument('--tomi_file', type=str, default='rephrased_tomi',
                       help='Path to rephrased TOMI test JSONL file')
    parser.add_argument('--train_size', type=int, default=800,
                       help='Number of training examples')
    parser.add_argument('--test_size', type=int, default=200,
                       help='Number of test examples')
    parser.add_argument('--template_type', type=str, default='base',
                       help='Template type for prompts')
    args = parser.parse_args()

    # Load existing train and test datasets
    train_dataset = Dataset.from_json(os.path.join(args.local_dir, f'{args.tomi_file}_train.jsonl'))
    test_dataset = Dataset.from_json(os.path.join(args.local_dir, f'{args.tomi_file}_test.jsonl'))
    
    # Map the processing function over the datasets
    train_dataset = train_dataset.map(
        lambda x, i: process_tomi_data(x, i, 'train', args.template_type),
        with_indices=True
    )
    test_dataset = test_dataset.map(
        lambda x, i: process_tomi_data(x, i, 'test', args.template_type),
        with_indices=True
    )

    # Save datasets
    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)
    breakpoint()
    
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)

if __name__ == '__main__':
    main() 