#!/bin/bash

#SBATCH -A2TExpt                              # Job name
#SBATCH -N1 --gres=gpu:RTX_6000:2 --ntasks-per-node=10   # Number of nodes, GPUs, and cores required
#SBATCH --mem-per-gpu=20G                           # Memory per gpu
#SBATCH -t720                               # Duration of the job (Ex: 15 mins)
#SBATCH -oReport-%j.out                             # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL                  # Mail preferences
#SBATCH --mail-user=asarswat8@gatech.edu           # e-mail address for notifications

cd $HOME/scratch/NLP/ArXiv_NLP                          # Change to working directory created in $HOME

module load anaconda3                  # Load module dependencies
conda activate eml
srun python3 train_bart.py