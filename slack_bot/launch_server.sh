#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --time=12:00:00
#SBATCH --output=/home/m/mehrad/brikiyou/scratch/spock/slack_bot/out/slurm-%j.out
#SBATCH --error=/home/m/mehrad/brikiyou/scratch/spock/slack_bot/out/slurm-%j.err


module load BalamEnv
source /home/m/mehrad/brikiyou/scratch/new_spock_venv/bin/activate
python3 /home/m/mehrad/brikiyou/scratch/spock/slack_bot/server.py
rm -f /home/m/mehrad/brikiyou/scratch/spock/slack_bot/slurm*