export N_GPUS=2
export BASE_MODEL=./models/Qwen2.5-1.5B
export DATA_DIR=./data
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-1.5b
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh