#!/bin/bash
#SBATCH -J inpaint-epic
#SBATCH --comment "inpaint-epic"

#SBATCH --qos level0
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --gres=gpu
#SBATCH --partition=gpu
# SBATCH -w lambda-hyperplane01
#SBATCH --account=research
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00

#SBATCH -o "/scratch/generalvision/EgoCentric/outputs/inpaint_slurm/%x.%j.out"
#SBATCH -e "/scratch/generalvision/EgoCentric/outputs/inpaint_slurm/%x.%j.err"

START_PART=$1
END_PART=$2
HOME=/home/lipuhao

echo "running slrum file"

for ((p=$START_PART; p<=$END_PART; p++))
do
    echo "running mask file with part: $p"
    python inpaint_part.py --part $p --batch_size 8 --ckpt ./release_model/E2FGVI-HQ-CVPR22.pth --model 'e2fgvi_hq' --basedir /scratch/generalvision/EgoCentric/EPIC-KITCHEN
done

echo "exit slrum file" 