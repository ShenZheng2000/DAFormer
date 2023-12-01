# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

#!/bin/bash

# NOTE: change latest.pth to best (or other) checkpoint if needed

TEST_ROOT=$1
CONFIG_FILE="${TEST_ROOT}/*${TEST_ROOT: -1}.json"
CHECKPOINT_FILE="${TEST_ROOT}/best_mIoU_iter_*.pth"
SHOW_DIR="${TEST_ROOT}/preds/"
echo 'Config File:' $CONFIG_FILE
echo 'Checkpoint File:' $CHECKPOINT_FILE
echo 'Predictions Output Directory:' $SHOW_DIR
python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU --show-dir ${SHOW_DIR} --opacity 1
