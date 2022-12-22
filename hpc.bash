#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time 02:10:00
#SBATCH --mem=16gb
#SBATCH --ntasks-per-node 10


#< Charge resources to account
#SBATCH --reservation=hpc_t_2022_dlagm20221214    
#SBATCH --account T_2022_dlagm

echo $SLURM_JOB_NODELIST

echo  #OMP_NUM_THREADS : $OMP_NUM_THREADS

module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate pytorch-cuda-11.1

python ./main.py --training_set_path /hpc/archive/T_2022_DLAGM/alex.ergasti/UTKFace/train \
        --checkpoint_path /hpc/group/T_2022_DLAGM/alex.ergasti/models \
        --validation_set_path /hpc/archive/T_2022_DLAGM/alex.ergasti/UTKFace/val \
        --test_set_path /hpc/archive/T_2022_DLAGM/alex.ergasti/UTKFace/test \


conda deactivate
