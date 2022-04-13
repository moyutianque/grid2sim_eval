work_path=$(dirname $0)
now=$(date +%s)
split='val_unseen'

PYTHONPATH=./ python -u run.py --exp-config config/no_learning.yaml \
TASK_CONFIG.DATASET.DATA_PATH "data/mln_v1/annt/{split}/{split}.json.gz" \
TASK_CONFIG.TASK.NDTW.GT_PATH "data/mln_v1/annt/{split}/{split}_gt.json.gz" \
TASK_CONFIG.DATASET.SPLIT $split \
EVAL.NONLEARNING.RESULT_PATH  $work_path/$split\_grid_pred_results.json \
EVAL.NONLEARNING.DUMP_DIR $work_path \
EVAL.SPLIT $split
#\
#2>&1 | tee $work_path/train.$now.log.out