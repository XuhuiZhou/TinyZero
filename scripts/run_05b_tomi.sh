export N_GPUS=2
export BASE_MODEL=/usr2/xuhuiz/models/Qwen2.5-0.5B
export DATA_DIR=./data/tomi_data
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=tomi-qwen2.5-0.5b-trial
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero_tomi.sh