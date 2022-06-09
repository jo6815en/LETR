#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 8
#SBATCH --time 1-00:00:00
#SBATCH --output /proj/feature_detection/users/%u/logs/%j.out
#SBATCH -A berzelius-2022-120
#

echo ""
echo "This job was started as: python3 -u $@"
echo ""

singularity exec --nv --bind /proj/feature_detection/users/$USER:/workspace \
  --bind /proj/feature_detection/data:/my_data \
  --pwd /workspace/LETR/ \
  --env PYTHONPATH=/workspace/LETR/ \
  /proj/feature_detection/letr.sif \
  python3 -u -m torch.distributed.launch --nproc_per_node=8 --master_port=$RANDOM \
  $@ --distributed

#
#EOF
