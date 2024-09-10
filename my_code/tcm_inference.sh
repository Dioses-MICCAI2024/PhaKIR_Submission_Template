CONFIG_PATH="my_code/tapis/TCM_CONFIG.yaml"
CHECK_POINT="my_model/tcm_model.pyth"
#-------------------------

TEST_DIR='inputs'
OUTPUT_DIR='outputs'

BATCH=1
WORKERS=10

#python III_multitask/multitask_recognition/format_videos.py $TEST_DIR $OUTPUT_DIR

python -W ignore my_code/tapis/run_net_tcm.py \
--cfg $CONFIG_PATH \
DATA_LOADER.NUM_WORKERS $WORKERS \
TEST.BATCH_SIZE $BATCH \
TEST.CHECKPOINT_FILE_PATH $CHECK_POINT \
OUTPUT_DIR $OUTPUT_DIR