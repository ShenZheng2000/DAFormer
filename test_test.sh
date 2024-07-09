# Purpose: get test predictions for public leaderboard
TEST_ROOT=$1
CONFIG_FILE="${TEST_ROOT}/*${TEST_ROOT: -1}.json"
CHECKPOINT_FILE="${TEST_ROOT}/best_mIoU_iter_*.pth"
SAVE_PATH="${TEST_ROOT}/labelTrainIds" # NOTE: use for server submission

python -m tools.test \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    --test-set \
    --format-only \
    --eval-option imgfile_prefix=${SAVE_PATH} to_label_id=False