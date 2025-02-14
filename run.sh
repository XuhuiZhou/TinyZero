export N_GPUS=2
export BASE_MODEL=./models/Qwen2.5-3B
export DATA_DIR=./data
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=4f58c216e8b5d314ea2bab9409546321eaa683ef

bash ./scripts/train_tiny_zero.sh