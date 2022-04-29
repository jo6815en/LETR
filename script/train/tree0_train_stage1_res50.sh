# Fail the script if there is any failure
set -e

if [[ $# -eq 0 ]] ; then
    echo 'Require Experiment Name'
    exit 1
fi

# The name of this experiment.
name=$1

# Save logs and models under snap/gqa; make backup.
output=exp/$name
if [ -d "$output"  ]; then
    rm -rf $output
fi

mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

PYTHONPATH=$PYTHONPATH:./src python src/main.py --coco_path data/trees \
--output_dir $output --backbone resnet50 --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
--batch_size 1 --epochs 500 --lr_drop 200 --num_queries 1000  --num_gpus 1   --layer1_num 3 | tee -a $output/history.txt



