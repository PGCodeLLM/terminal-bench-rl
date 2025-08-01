#!/bin/bash
set -x


export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export PYTHONPATH="$(pwd):$(pwd)/external/rllm"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1 
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000


HF_MODEL_NAME=${HF_MODEL_NAME:-"Qwen/Qwen3-32B"}

# Training configuration
MODEL_PATH=${MODEL_PATH:-"./models/${HF_MODEL_NAME}"} 

DATA_DIR=${DATA_DIR:-"./data"}
PROJECT_NAME=${PROJECT_NAME:-"terminal_bench_grpo_agent"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"qwen3-32b_terminal_agent"}

MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-32768}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-6000}
# Calculate max response length as sequence length minus prompt length
MAX_TOKENS_ALLOWED_FOR_MODEL_IN_TRAJECTORY=$((MAX_SEQUENCE_LENGTH - MAX_PROMPT_LENGTH))

NUM_EPOCHS=${NUM_EPOCHS:-1}  # Number of epochs to train

# GRPO-specific parameters
N_ROLLOUTS=${N_ROLLOUTS:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}     
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1}   
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1} 

# GPU configuration
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
NNODES=${NNODES:-1}
TP_SIZE=${TP_SIZE:-2}  # Tensor parallel size for vLLM

# Sequence parallelism for long sequences (FSDP only)
ULYSSES_SEQUENCE_PARALLEL_SIZE=${ULYSSES_SEQUENCE_PARALLEL_SIZE:-4}

# Learning rates
ACTOR_LR=${ACTOR_LR:-1e-6}

# LoRA is not supported - removed configuration

# Agent/Environment configuration
MAX_STEPS=${MAX_STEPS:-50}
TRAJECTORY_TIMEOUT=${TRAJECTORY_TIMEOUT:-600}  # 10 minutes per trajectory

VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.6}  # GPU memory utilization for vLLM

SAVE_FREQ=${SAVE_FREQ:-25}  # Save frequency in epochs
REJECTION_SAMPLING_MULTIPLIER=${REJECTION_SAMPLING_MULTIPLIER:-2}  

# Run training directly (mappings already patched)
python3 -m rllm.trainer.verl.train_agent_ppo \
    algorithm.adv_estimator=loop \
    data.train_files=$DATA_DIR/train.parquet \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_files=$DATA_DIR/val.parquet \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_TOKENS_ALLOWED_FOR_MODEL_IN_TRAJECTORY \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    env.name=terminal_bench \
    +env.env_args.no_rebuild=False \
    +env.env_args.timeout=$TRAJECTORY_TIMEOUT \
    agent.max_steps=$MAX_STEPS \
    agent.name=terminal_bench_agent \
    agent.async_engine=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_shm=False \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32000 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ULYSSES_SEQUENCE_PARALLEL_SIZE \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$VLLM_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.n=$N_ROLLOUTS \
    actor_rollout_ref.rollout.temperature=1.2 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.max_model_len=$MAX_ROLLOUT_LENGTH \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.chat_scheduler=verl.schedulers.naive_chat_scheduler.NaiveChatCompletionScheduler \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.mask_truncated_samples=False \
    algorithm.gamma=0.99 \
    algorithm.lam=0.95 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=-1 \
    trainer.total_epochs=$NUM_EPOCHS \
    trainer.val_before_train=False \
    trainer.rejection_sample=True \
    trainer.rejection_sample_multiplier=$REJECTION_SAMPLING_MULTIPLIER \
    "$@"