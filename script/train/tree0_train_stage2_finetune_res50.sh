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
if [ ! -d "$output"  ]; then
    echo "folder not exist"
    mkdir -p $output/src
    cp -r src/* $output/src/
    cp $0 $output/run.bash

    PYTHONPATH=$PYTHONPATH:./src python src/main.py --coco_path data/trees \
    --output_dir $output  --LETRpost  --backbone resnet50  --layer1_frozen --resume exp/tisdag_test3_stage2/checkpoints/checkpoint.pth  \
    --no_opt --batch_size 1  --epochs 25  --lr_drop 25  --num_queries 1000  --num_gpus 1  --lr 1e-5  --label_loss_func focal_loss \
    --label_loss_params '{"gamma":2.0}'  --save_freq 1  |  tee -a $output/history.txt 


else
    echo "folder already exist"
fi
