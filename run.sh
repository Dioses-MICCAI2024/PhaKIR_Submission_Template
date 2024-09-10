TEST_DIR='inputs'
OUTPUT_DIR='outputs'

export TEST_DIR='inputs'
export OUTPUT_DIR='outputs'

BATCH=1
WORKERS=10

mkdir -p $OUTPUT_DIR
# chmod 777 $OUTPUT_DIR

python format_videos.py

sh ./my_code/must_inference.sh

sh ./my_code/tcm_inference.sh

#sh ./my_code/segmentation_inference.sh