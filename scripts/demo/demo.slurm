#!/bin/bash
#SBATCH -J e2fgvi_test
#SBATCH --comment "e2fgvi demo test"
#SBATCH -p gpu
#SBATCH --qos level0
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH -t 00-01:00:00
#SBATCH --account research

#SBATCH -o "/scratch/lipuhao/dev/MAH/DataPreprocess/E2FGVI/outputs/slurm/%x.%j.out"
#SBATCH -e "/scratch/lipuhao/dev/MAH/DataPreprocess/E2FGVI/outputs/slurm/%x.%j.err"


echo "run the test slurm"
python test_slurm.py
python test.py --model e2fgvi --video examples/tennis --mask examples/tennis_mask --ckpt release_model/E2FGVI-CVPR22.pth
