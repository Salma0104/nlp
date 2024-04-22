#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=82G

# Email notifications 
#SBATCH --mail-type=END
#SBATCH --mail-user=shassan6@sheffield.ac.uk
# Change the name of the output log file.
#SBATCH --output=./logs/o_%j.out
#SBATCH --error=./logs/e_%j.err
# Request 8 hours
#SBATCH --time=8:00:00

# cd logs
# shopt -s extglob
# rm -v !(*$SLURM_JOB_ID*)
module load Anaconda3/2022.05
source activate nlp
python run_experiment.py --access_token=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
