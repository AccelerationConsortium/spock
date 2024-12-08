#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --time=1:00:00


module load BalamEnv
source /home/m/mehrad/brikiyou/scratch/new_spock_venv/bin/activate
python3 /home/m/mehrad/brikiyou/scratch/spock/slack_bot/server.py
rm -f /home/m/mehrad/brikiyou/scratch/spock/slack_bot/slurm*