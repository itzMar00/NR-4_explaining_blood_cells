#!/bin/bash

# ============ README ================== #
# This script is a default setting for running a small train dataset on WBCatt with only cell types as labels
# The paths used in the script are likely not automated nor specified to your path.
# Usage: cd into project root (NR-4 directory)
# then run sbatch scripts/<filename>.sh
# * Make a copy and change the paths to match

# SOME USEFUL SLURM: squeue, squeue --start (estimate start time), sacct (finished jobs), sacct -j jobid (info of job)
# scancel jobid, scancel -u myusername
# tail -f outputfile.out to watch live output, cat outputfile.out to view current file
# sbatch --dependency=afterany:<jobid> second_job.sh (execute second_job after <jobid> done running
# sbatch --mem=1G job.sh will override whatever --mem settings the job.sh file has
# ============= END readme ============= #

#SBATCH --partition dllabdlc_gpu-rtx2080
#SBATCH --job-name sg2ada_wbc_m500
#SBATCH --output run_logs/%x-%A.out
#SBATCH --error run_logs/%x-%A.err

#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00



PROJECT_DIR=$PWD  # Project dir where train.py is
RESULT_DIR=$PROJECT_DIR/training-runs/stylegan3-r/results # specify output results dir for stylegan's train.py to generate
DATA_PATH=$PROJECT_DIR/datasets/wbc-256x256_m500.zip # path to dataset



echo "Workingdir: $PWD";
echo "=============================="
echo "Running training"
echo "Start Time: $(date)"
echo "Hostname: $(hostname)"
echo "Running job $SLURM_JOB_NAME"
echo "using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"
echo "DATA_PATH = $DATA_PATH"
echo "RESULT_DIR = $RESULTS_DIR"
echo "=============================="

mkdir -p run_logs

# Configure wandb
source ~/wandb_config.sh
# export WANDB_PROJECT=""
export WANDB_NAME="sg2ada-m500-$(date +%Y%m%d-%H%M%S)"
export WANDB_TAGS="m500,${SLURM_JOB_ID}"
export WANDB_MODE="online"


# Activate env
source ~/miniconda3/bin/activate
conda activate /work/dlclarge2/tranh-nr4/environment

python $PROJECT_DIR/NR-4_explaining_blood_cells/train.py  \
    --outdir=$RESULT_DIR \
    --data=$DATA_PATH \
    --cfg=stylegan2 --gpus=1 --batch=16 --gamma=0.8192 \
    --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384 \
    --cond=True --aug=ada --snap 10

echo "=== JOB FINISHED ==="
echo "At: $(date)"
