#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=0
#SBATCH --output=/home/m/mehrad/brikiyou/scratch/slurm-%j.out
#SBATCH --error=/home/m/mehrad/brikiyou/scratch/slurm-%j.err

module load BalamEnv
module load python/3.8
source /home/m/mehrad/brikiyou/scratch/new_spock_venv/bin/activate

cd /home/m/mehrad/brikiyou/scratch/spock/slack_bot

python3 /home/m/mehrad/brikiyou/scratch/spock/slack_bot/scripts/process_generate_podcast.py \
    --paper "/home/m/mehrad/brikiyou/scratch/spock/slack_bot/papers/Artificial_Intelligence-Enabled_Optimization_of_Battery-Grade_Lithium_Carbonate_Production.pdf" \
    --user_id "U073D78M8UT" \
    --channel_id "D073ZCA5L6N" \
    --initial_comment "Here's the audio podcast for your pdf!"
