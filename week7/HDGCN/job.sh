#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH --mem 1000 # 2GB solicitados.
#SBATCH -p mlow # or mlow Partition to submit to master low prioriy queue
#SBATCH --gres gpu:1 # Para pedir Pascales MAX 8
#SBATCH -o logs/%x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e logs/%x_%u_%j.err # File to which STDERR will be written
python main.py --config ./config/hmdb51/bone_com_1.yaml