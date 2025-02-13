#!/bin/bash

python3 -m verl.trainer.main_generation \
model.path=/usr2/xuhuiz/models/Qwen2.5-0.5B/checkpoints/TinyZero/countdown-qwen2.5-0.5b/actor/global_step_200 \
data.path=./data/test.parquet \
data.output_path=./data/test_generated.parquet \
data.prompt_key=prompt \
data.n_samples=1 \
data.batch_size=1 \
trainer.n_gpus_per_node=1 \
trainer.nnodes=1 \
rollout.temperature=1.0 \
rollout.top_k=-1 \
rollout.top_p=0.7 \
rollout.prompt_length=256 \
rollout.response_length=1024 \
rollout.tensor_model_parallel_size=1 \
rollout.gpu_memory_utilization=0.4 \
rollout.micro_batch_size=256 \
rollout.log_prob_micro_batch_size=8 