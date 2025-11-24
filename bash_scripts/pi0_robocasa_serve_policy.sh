HOME_DIR=$(pwd)
BASE_DIR=/home/changyeon/workspace
CONDA_PATH=/home/changyeon/miniconda3

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_FLAGS="--xla_gpu_deterministic_ops=true" #NOTE : 재현성 위해 무조건 필요!

# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# export JAX_TRACEBACK_FILTERING=off 

unset LEROBOT_HOME



CKPT_PATH="/home/changyeon/ckpts"
CONFIG_NAME="pi05_robocasa_100demos_base"
EXP_NAME="pi05_robocasa_as50_jax"

POLICY_SEED=42
CKPT_STEP=30000

DEBUG=0   # 1 to return top-10 dists per step

POLICY_DIRS=(
    # "$CKPT_PATH/$CONFIG_NAME/$EXP_NAME/$CKPT_STEP"
)

RANDOM_PORT=39281

echo "=== Configuration ==="
echo "ENV_SEED: 42 "
echo "PORT: $RANDOM_PORT"

SERVE_CMD="uv run $BASE_DIR/custom_openpi/scripts/serve_policy.py --port=$RANDOM_PORT \
    policy:checkpoint \
    --policy.config=$CONFIG_NAME \
    --policy.dir ${POLICY_DIRS[@]}"

echo "Starting policy server with command:"
echo "$SERVE_CMD"
eval $SERVE_CMD