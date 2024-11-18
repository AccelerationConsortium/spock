#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --time=24:00:00


module load BalamEnv

python3 /home/m/mehrad/brikiyou/scratch/spock_package/spock/slack_bot/server.py

rm -rf slurm*
