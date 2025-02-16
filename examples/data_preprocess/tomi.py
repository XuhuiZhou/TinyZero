"""
Preprocess dataset for TOMI test - Theory of Mind reasoning task
"""

import os
import json
from datasets import Dataset
from typing import List
from verl.utils.hdfs_io import copy, makedirs
import argparse
from huggingface_hub import hf_hub_download, login


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
    parser.add_argument('--hf_token', type=str, default=None,
                       help='Hugging Face authentication token')
    args = parser.parse_args()

    # Create local directory if it doesn't exist
    os.makedirs(args.local_dir, exist_ok=True)

    # Authenticate with Hugging Face if token is provided
    if args.hf_token:
        login(token=args.hf_token)
    elif os.getenv("HUGGINGFACE_TOKEN"):
        login(token=os.getenv("HUGGINGFACE_TOKEN"))
    else:
        print("Warning: No Hugging Face token provided. Set HF_TOKEN environment variable or use --hf_token")
        print("You can find your token at https://huggingface.co/settings/tokens")

    # Check if TOMI files exist, if not download from Huggingface
    train_file = os.path.join(args.local_dir, f'{args.tomi_file}_train.jsonl')
    test_file = os.path.join(args.local_dir, f'{args.tomi_file}_test.jsonl')
    
    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        print("Downloading TOMI dataset from Huggingface...")
        repo_id = "Xuhui/social-reasoning"
        train_filename = "rephrased_tomi_train.jsonl"
        test_filename = "rephrased_tomi_test.jsonl"
        downloaded_train_file = hf_hub_download(
            repo_id=repo_id,
            filename=train_filename,
            repo_type="dataset"
            
        )
        downloaded_test_file = hf_hub_download(
            repo_id=repo_id,
            filename=test_filename,
            repo_type="dataset"
        )
        # Load and split the downloaded data
        with open(downloaded_train_file, 'r') as f:
            train_data = [json.loads(line) for line in f]
        with open(downloaded_test_file, 'r') as f:
            test_data = [json.loads(line) for line in f]
        
        # write train and test data to local file
        with open(train_file, 'w') as f:
            for item in train_data:
                f.write(json.dumps(item) + '\n')
        with open(test_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')

    # Load and process datasets
    train_dataset = Dataset.from_json(train_file)
    test_dataset = Dataset.from_json(test_file)
    
    # Map the processing function over the datasets
    train_dataset = train_dataset.map(
        lambda x, i: process_tomi_data(x, i, 'train', args.template_type),
        with_indices=True
    )
    test_dataset = test_dataset.map(
        lambda x, i: process_tomi_data(x, i, 'test', args.template_type),
        with_indices=True
    )

    # Save processed datasets
    train_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))

    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)

if __name__ == '__main__':
    main() 