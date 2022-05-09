work_path=$(dirname $0)
now=$(date +%s)
split='val_seen'

PYTHONPATH=./ python -u run.py --exp-config config/no_learning.yaml \
TASK_CONFIG.DATASET.DATA_PATH "data/mln3d/annt/{split}/{split}.json.gz" \
TASK_CONFIG.TASK.NDTW.GT_PATH "data/mln3d/annt/{split}/{split}_gt.json.gz" \
EVAL.DATA_PATH "data/mln3d/annt/{split}/{split}.json.gz" \
TASK_CONFIG.DATASET.SPLIT $split \
EVAL.NONLEARNING.AGENT ActionSimAgent \
EVAL.NONLEARNING.RESULT_PATH  $work_path/$split\_mln3d_routes.json \
EVAL.NONLEARNING.DUMP_DIR $work_path \
EVAL.SPLIT $split
#\
#2>&1 | tee $work_path/train.$now.log.out