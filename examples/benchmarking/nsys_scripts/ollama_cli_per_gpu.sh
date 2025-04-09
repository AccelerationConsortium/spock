#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=2:00:00
#SBATCH --output=/home/m/mehrad/brikiyou/scratch/spock/slack_bot/out/slurm-%j.out
#SBATCH --error=/home/m/mehrad/brikiyou/scratch/spock/slack_bot/out/slurm-%j.err
#SBATCH --mail-user=youssef.briki@umontreal.ca
#SBATCH --mail-type=FAIL 
#SBATCH -p compute_full_node

