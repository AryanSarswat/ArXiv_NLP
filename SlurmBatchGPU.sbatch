#!/bin/bash

#SBATCH -JComputeMetric                              # Job name
#SBATCH -N1 --gres=gpu:1 -C A100-40GB --ntasks-per-node=20   # Number of nodes, GPUs, and cores required
#SBATCH -t960                               # Duration of the job (Ex: 15 mins)
#SBATCH -oReport-%j.out                             # Combined output and error messages file
#SBATCH --mail-type=END,FAIL                  # Mail preferences
#SBATCH --mail-user=asarswat8@gatech.edu           # e-mail address for notifications

cd $HOME/scratch/NLP/ArXiv_NLP                          # Change to working directory created in $HOME

module load anaconda3                  # Load module dependencies
conda activate eml
python3 compute_metric.py base
