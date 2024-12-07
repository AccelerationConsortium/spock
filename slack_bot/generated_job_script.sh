#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:05:00
#SBATCH --gpus-per-node=1
#SBATCH --output=/home/m/mehrad/brikiyou/scratch/slurm-%j.out
#SBATCH --error=/home/m/mehrad/brikiyou/scratch/slurm-%j.err


module load BalamEnv
module load python/3.8

source /home/m/mehrad/brikiyou/scratch/new_spock_venv/bin/activate

if [[ "" == "llama" ]]; then
    source /home/m/mehrad/brikiyou/scratch/to_run.sh
    ollama serve > /home/m/mehrad/brikiyou/scratch/ollama.log 2>&1 &
fi

python3 /home/m/mehrad/brikiyou/scratch/spock/slack_bot/scripts/process_publication.py     --publication "https://www.nature.com/articles/s41467-023-44599-9" \
    --questions "" \
    --user_id "U073D78M8UT" \
    --channel_id "D073ZCA5L6N"
