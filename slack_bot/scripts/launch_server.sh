#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --time=24:00:00
#SBATCH --output=/home/m/mehrad/brikiyou/scratch/spock/slack_bot/out/slurm-%j.out
#SBATCH --error=/home/m/mehrad/brikiyou/scratch/spock/slack_bot/out/slurm-%j.err

module load BalamEnv
module load python
source /home/m/mehrad/brikiyou/scratch/new_spock_venv_2/bin/activate

timeout 23h python3 /home/m/mehrad/brikiyou/scratch/spock/slack_bot/server.py
exit_code=$?

if [[ $exit_code -eq 124 ]]; then
    echo "$(date): Job timed out. Resubmitting..."
    sbatch "$0"
else
    echo "$(date): Job completed or stopped before timeout."
fi
