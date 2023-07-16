#!/bin/bash

EXP_NAME=$1

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/../..")

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
export PYTHONPATH=$PROJECT_DIR
export PYTHONPATH=$PYTHONPATH:$HOME/offline_c_learning/bullet-manipulation
export PYTHONPATH=$PYTHONPATH:$HOME/offline_c_learning/bullet-manipulation/roboverse/envs/assets/bullet-objects
export PYTHONPATH=$PYTHONPATH:$HOME/offline_c_learning/multiworld
export PATH=$PATH:$HOME/anaconda3/envs/stable_contrastive_rl/bin
export LOG_ROOT="/projects/rsalakhugroup/chongyiz"

export EVAL_SEED=31

declare -a seeds=(0 1)

for seed in "${seeds[@]}"; do
  export CUDA_VISIBLE_DEVICES=$seed
  rm -r "$LOG_ROOT"/offline_c_learning/td_cpc_logs/env6_250k/"${EVAL_SEED}"/"${EXP_NAME}"/run"$seed"/id0
  mkdir -p "$LOG_ROOT"/offline_c_learning/td_cpc_logs/env6_250k/"${EVAL_SEED}"/"${EXP_NAME}"/run"$seed"/id0
  nohup \
  python experiments/train_eval_stable_contrastive_rl.py \
    --local \
    --gpu \
    --data_dir "$LOG_ROOT"/offline_c_learning/dataset \
    --base_log_dir "$LOG_ROOT"/offline_c_learning/td_cpc_logs/env6_250k/"${EVAL_SEED}" \
    --name "${EXP_NAME}" \
    --run_id "$seed" \
    --arg_binding method_name=td_cpc \
    --arg_binding eval_seeds="${EVAL_SEED}" \
    --arg_binding trainer_kwargs.augment_probability=0.25 \
    --arg_binding trainer_kwargs.use_td=False \
    --arg_binding trainer_kwargs.use_td_cpc=True \
    --arg_binding num_demos=18 \
  > "$LOG_ROOT"/offline_c_learning/td_cpc_logs/env6_250k/"${EVAL_SEED}"/"${EXP_NAME}"/run"$seed"/id0/stream.log 2>&1 & \
  sleep 2
done
