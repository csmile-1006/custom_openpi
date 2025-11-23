HOME_DIR=$(pwd)
BASE_DIR=/home/changyeon/workspace
CONDA_PATH=/home/changyeon/miniconda3

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_FLAGS="--xla_gpu_deterministic_ops=true" #NOTE : 재현성 위해 무조건 필요!

# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# export JAX_TRACEBACK_FILTERING=off 

unset LEROBOT_HOME



CKPT_PATH="/home/changyeon/ckpts/"
CONFIG_NAME="pi0_robocasa_100demos_base"

POLICY_SEED=42
CKPT_STEP=59999

DEBUG=0   # 1 to return top-10 dists per step

POLICY_DIRS=(
    "$CKPT_PATH/$CONFIG_NAME/$CKPT_STEP"
)

RANDOM_PORT=39281
SEEDS=(
  "0"
  "42"
  "123"
)

TASK_NAMES=(
#   "TurnSinkSpout"
#   "TurnOnStove"
#   "TurnOnSinkFaucet"
#   "TurnOnMicrowave"
  "TurnOffStove"
#   "TurnOffSinkFaucet"
#   "TurnOffMicrowave"
#   "PnPStoveToCounter" #7
#   "PnPSinkToCounter"
  "PnPMicrowaveToCounter"
#   "PnPCounterToStove"
#   "PnPCounterToSink"
  "PnPCounterToMicrowave"
#   "PnPCounterToCab"
#   "PnPCabToCounter"
#   "OpenSingleDoor"
#   "OpenDrawer"
#   "OpenDoubleDoor"
  "CoffeeSetupMug"
#   "CoffeeServeMug"
#   "CoffeePressButton"
#  "CloseSingleDoor"
#   "CloseDrawer"
#   "CloseDoubleDoor" # 23
) # 24 tasks in total

for TASK_ID in "${!TASK_NAMES[@]}"; do
  for SEED_ID in "${!SEEDS[@]}"; do
    TASK_NAME="${TASK_NAMES[$TASK_ID]}"
    SEED=${SEEDS[$SEED_ID]}
    OUTPUT_DIR="$BASE_DIR/output/openpi/robocasa_100/$CONFIG_NAME/checkpoint-$CKPT_STEP/$TASK_NAME-seed$SEED"
    mkdir -p "$OUTPUT_DIR"

    echo "POLICY : ${POLICY_DIRS[@]}  | TASK_NAME: ${TASK_NAME} | SEED: ${SEED} | PORT: ${RANDOM_PORT}"

    EVAL_CMD="python $BASE_DIR/custom_openpi/examples/robocasa/scripts/robocasa_eval.py \
        --args.port=$RANDOM_PORT \
        --args.seed=$SEED \
        --args.env_name \"$TASK_NAME\" \
        --args.video-dir \"$OUTPUT_DIR\" \
        --args.n-episodes=50 \
        --args.generative_textures"

    echo "Starting evaluation with command:"
    echo "$EVAL_CMD"
    eval $EVAL_CMD
    done
done
 