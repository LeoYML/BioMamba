#!/bin/bash

# Define different sets of hyperparameters
LR_MAX_VALUES=(1e-3 1e-4 1e-5)
LR_MIN_VALUES=(1e-4 1e-5 1e-6)
NUM_TRAINING_STEPS_VALUES=(20 30 60)
WEIGHT_DECAY_VALUES=(0.1 0.1 0.1)
CUDA_DEVICES=(0 1 2)

# Define log files for each run
LOG_FILES=(
"logs/rough_inesearch_1e-3_1e-4.log"
"logs/rough_inesearch_1e-4_1e-5.log"
"logs/rough_inesearch_1e-5_1e-6.log"
)

# Iterate over the set of hyperparameters
for i in "${!LR_MAX_VALUES[@]}"; do
    # Set the hyperparameters for the current run
    LR_MAX=${LR_MAX_VALUES[$i]}
    LR_MIN=${LR_MIN_VALUES[$i]}
    NUM_TRAINING_STEPS=${NUM_TRAINING_STEPS_VALUES[$i]}
    WEIGHT_DECAY=${WEIGHT_DECAY_VALUES[$i]}
    CUDA_DEVICE=${CUDA_DEVICES[$i]}
    LOG_FILE=${LOG_FILES[$i]}

    # Run the Python script with the current hyperparameters and assign to the specified CUDA device
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE nohup python3 hp_tune.py \
        --lr_max $LR_MAX \
        --lr_min $LR_MIN \
        --num_training_steps $NUM_TRAINING_STEPS \
        --weight_decay $WEIGHT_DECAY \
        > $LOG_FILE 2>&1 &

    echo "Started hyperparameter tuning with lr_max=$LR_MAX, lr_min=$LR_MIN, num_training_steps=$NUM_TRAINING_STEPS, weight_decay=$WEIGHT_DECAY on CUDA device $CUDA_DEVICE"
done

echo "All hyperparameter tuning jobs have been started."
