#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --time=12:00:00
#SBATCH --output=/home/m/mehrad/brikiyou/scratch/spock/slack_bot/out/slurm-%j.out
#SBATCH --error=/home/m/mehrad/brikiyou/scratch/spock/slack_bot/out/slurm-%j.err


module load BalamEnv
module load python
source /home/m/mehrad/brikiyou/scratch/new_spock_venv_2/bin/activate
python3 /home/m/mehrad/brikiyou/scratch/spock/slack_bot/server.py
