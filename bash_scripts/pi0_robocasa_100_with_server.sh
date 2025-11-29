HOME_DIR=$(pwd)
BASE_DIR=/home/changyeon/workspace
CONDA_PATH=/home/changyeon/miniconda3
SAVE_DIR="/home/ubuntu/data/changyeon/"
CKPT_PATH="/home/ubuntu/data/changyeon/ckpts"

CONFIG_NAME=${1:-"pi05_robocasa_100demos_base"}
EXP_NAME=${2:-"pi05_robocasa_as50_jax"}
RANDOM_PORT=${3:-39281}

CKPT_STEPS=(
  "59999"
  "50000"
  "40000"
  "30000"
  "20000"
  "10000"
)
export XLA_FLAGS="--xla_gpu_deterministic_ops=true" #NOTE : 재현성 위해 무조건 필요!

# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# export JAX_TRACEBACK_FILTERING=off 

unset LEROBOT_HOME

# Function to cleanup background process
cleanup() {
    if [ ! -z "$SERVER_PID" ]; then
        echo "Killing policy server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
        echo "Policy server terminated."
    fi
}

# Set trap to cleanup on script exit (handles Ctrl+C, etc.)
trap cleanup EXIT INT TERM

for CKPT_STEP in "${CKPT_STEPS[@]}"; do
  POLICY_SEED=42
  DEBUG=0   # 1 to return top-10 dists per step

  POLICY_DIRS=(
      "$CKPT_PATH/$CONFIG_NAME/$EXP_NAME/$CKPT_STEP"
  )

  echo "=== Configuration ==="
  echo "CONFIG_NAME: $CONFIG_NAME"
  echo "EXP_NAME: $EXP_NAME"
  echo "CKPT_STEP: $CKPT_STEP"
  echo "PORT: $RANDOM_PORT"
  echo "POLICY_DIRS: ${POLICY_DIRS[@]}"

  # Start policy server in background
  echo "=== Starting Policy Server ==="
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
  SERVE_CMD="uv run $BASE_DIR/custom_openpi/scripts/serve_policy.py --port=$RANDOM_PORT \
      policy:checkpoint \
      --policy.config=$CONFIG_NAME \
      --policy.dir ${POLICY_DIRS[@]}"

  echo "Starting policy server with command:"
  echo "$SERVE_CMD"
  eval $SERVE_CMD &
  SERVER_PID=$!

  # Wait a bit for server to start
  echo "Waiting for policy server to start..."
  sleep 5

  # Check if server is still running
  if ! kill -0 $SERVER_PID 2>/dev/null; then
      echo "Error: Policy server failed to start!"
      exit 1
  fi

  echo "Policy server started (PID: $SERVER_PID)"

  # Run evaluation loop
  echo "=== Starting Evaluation Loop ==="
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.1

  SEEDS=(
    "0"
    "42"
    "123"
  )

  TASK_NAMES=(
    "TurnOffStove"
    "PnPMicrowaveToCounter"
    "PnPCounterToMicrowave"
    "CoffeeSetupMug"
  ) # 24 tasks in total

  for TASK_ID in "${!TASK_NAMES[@]}"; do
    for SEED_ID in "${!SEEDS[@]}"; do
      TASK_NAME="${TASK_NAMES[$TASK_ID]}"
      SEED=${SEEDS[$SEED_ID]}
      OUTPUT_DIR="$SAVE_DIR/output/openpi/robocasa_100/$CONFIG_NAME/$EXP_NAME/$CKPT_STEP/$TASK_NAME-seed$SEED"
      mkdir -p "$OUTPUT_DIR"

      echo "POLICY : ${POLICY_DIRS[@]}  | TASK_NAME: ${TASK_NAME} | SEED: ${SEED} | PORT: ${RANDOM_PORT}"

      EVAL_CMD="/home/changyeon/miniconda3/envs/robocasa/bin/python $BASE_DIR/custom_openpi/examples/robocasa/scripts/robocasa_eval.py \
          --args.port=$RANDOM_PORT \
          --args.seed=$SEED \
          --args.env_name \"$TASK_NAME\" \
          --args.video-dir \"$OUTPUT_DIR\" \
          --args.n-episodes=1 \
          --args.generative_textures"

      echo "Starting evaluation with command:"
      echo "$EVAL_CMD"
      eval $EVAL_CMD
    done
  done

  echo "=== Evaluation Complete ==="
  
  # Explicitly cleanup before next iteration
  cleanup
  SERVER_PID=""  # Reset so trap won't try to kill non-existent process
  
  # Wait a bit before starting next server
  sleep 2

done