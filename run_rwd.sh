#!/bin/bash

set -e
cur_fname="$(basename $0 .sh)"
script_name=$(basename $0)

# Cluster parameters
partition=""
account=""

# Initialize TCP port and counter
TCP_INIT=28500
counter=0

# Declare arrays for different configurations
declare -a DATASETS=( "Medical" "Aquatic" "Game")
# declare -a DATASETS=("Game")
declare -a MODELS=("google/owlvit-base-patch16" "google/owlvit-large-patch14")
# declare -a NUM_SHOTS=(1 10 100)
declare -a NUM_SHOTS=(100)

# Declare associative array for CUR_INTRODUCED_CLS per dataset
declare -A CUR_INTRODUCED_CLS
CUR_INTRODUCED_CLS["Aerial"]=10
CUR_INTRODUCED_CLS["Surgical"]=6
CUR_INTRODUCED_CLS["Medical"]=6
CUR_INTRODUCED_CLS["Aquatic"]=4
CUR_INTRODUCED_CLS["Game"]=30

declare -A BATCH_SIZEs
BATCH_SIZEs["google/owlvit-base-patch16"]=48
BATCH_SIZEs["google/owlvit-large-patch14"]=24

declare -A IMAGE_SIZEs
IMAGE_SIZEs["google/owlvit-base-patch16"]=768
IMAGE_SIZEs["google/owlvit-large-patch14"]=840

# Before the for loop:
export CUDA_VISIBLE_DEVICES=0  # Only use GPU 1


for num_shot in "${NUM_SHOTS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    cur_cls=${CUR_INTRODUCED_CLS[$dataset]}
    for model in "${MODELS[@]}"; do
      BATCH_SIZE=${BATCH_SIZEs[$model]}
      IMAGE_SIZE=${IMAGE_SIZEs[$model]}
      tcp=$((TCP_INIT + counter))

      DATA_ROOT="$(realpath ../DATA)"

      echo -e "\n========== STARTING RUN: dataset=$dataset, shots=$num_shot, model=$model =========="

      start_train=$(date +%s)

      cmd="python main.py --model_name \"$model\" --num_few_shot $num_shot --batch_size $BATCH_SIZE \
      --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS $cur_cls --TCP $tcp --dataset $dataset \
      --data_root $DATA_ROOT --image_conditioned --image_resize $IMAGE_SIZE \
      --att_refinement --att_adapt --att_selection --use_attributes"

      echo -e "Executing:\n$cmd"
      eval "$cmd"

      end_train=$(date +%s)

      train_time=$((end_train - start_train))

      echo "Training+Testing time for dataset=$dataset, shots=$num_shot, model=$model: $train_time seconds" | tee -a timing_log.txt

      counter=$((counter + 1))
    done
  done
done
