#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --time 3-00:00:00
#SBATCH --output /proj/feature_detection/users/%u/logs/%j.out
#

echo ""
echo "This job was started as: python3 -u $@"
echo ""

singularity exec --nv --bind /proj/feature_detection/users/$USER:/workspace \
  --bind /proj/feature_detection/data:/my_data \
  --pwd /workspace/LETR/ \
  --env PYTHONPATH=/workspace/LETR/ \
  /proj/feature_detection/LETR.sif \
  python3 -u $@

#
#EOF
