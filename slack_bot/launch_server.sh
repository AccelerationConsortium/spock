#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH -p compute_full_node

source /home/m/mehrad/brikiyou/scratch/to_run.sh

source /home/m/mehrad/brikiyou/scratch/new_spock_venv/bin/activate

ollama serve > ollama.log 2>&1 &

python3 /home/m/mehrad/brikiyou/scratch/spock_package/spock/slack_bot/server.py

rm -rf slurm*
