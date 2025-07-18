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
#SBATCH --job-name stylex-2ada
#SBATCH --output logs/stylex-2ada/%x-%A.out
#SBATCH --error logs/stylex-2ada/%x-%A.err

#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00

PROJECT_DIR=$PWD  # Project dir or root folder where you call to this file sbatch /path/to/.sh
RESULT_DIR=$PROJECT_DIR/training-runs/ # specify output results dir for stylegan's train.py to generate
DATA_PATH=$PROJECT_DIR/datasets/wbc-256x256_full.zip # path to dataset
CLASSIFIER_PATH=$PROJECT_DIR/image_classification/resnet18_trained_model.pth
HOME_DIR=$HOME

source /etc/cuda_env
cuda12.6
which nvcc
echo "Workingdir: $PWD";
echo "=============================="
echo "Running StylEx-StyleGan2ADA Training"
echo "Start Time: $(date)"
echo "Hostname: $(hostname)"
echo "Running job $SLURM_JOB_NAME"
echo "using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"
echo "DATA_PATH = $DATA_PATH"
echo "RESULT_DIR = $RESULT_DIR"
echo "HOME_DIR =$HOME"
echo "=============================="

#mkdir -p


# Activate env
source ~/miniconda3/bin/activate
conda activate /work/dlclarge2/tranh-nr4/environment
source ~/wandb_config.sh
export WANDB_NAME="stylex-sg2ada-kicluster-$(date +%Y%m%d-%H%M%S)"
export WANDB_TAGS="${SLURM_JOB_ID},rtx2080"
export WANDB_MODE="online"
python $PROJECT_DIR/NR-4_explaining_blood_cells/train_stylex.py  \
    --outdir=$RESULT_DIR \
    --data=$DATA_PATH \
    --cfg=stylegan2 --gpus=1 --batch=16 --gamma=0.8192 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384 \
    --cond=True --aug=ada --snap 20 --classifier-path=$CLASSIFIER_PATH

echo "=== JOB FINISHED ==="
echo "At: $(date)"
