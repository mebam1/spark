NUM_MACHINES=1
# Auto-detect number of GPUs from CUDA_VISIBLE_DEVICES
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_LOCAL_GPUS=8  # Default to 8 if not set
else
    NUM_LOCAL_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
MACHINE_RANK=0

export WANDB_API_KEY="" # Modify this if you use wandb
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

accelerate launch \
    --num_machines $NUM_MACHINES \
    --num_processes $(( $NUM_MACHINES * $NUM_LOCAL_GPUS )) \
    --machine_rank $MACHINE_RANK \
    src/train_partcrafter_part.py \
        --pin_memory \
        --allow_tf32 \
$@
