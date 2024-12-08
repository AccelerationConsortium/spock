#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/m/mehrad/brikiyou/scratch/slurm-%j.out
#SBATCH --error=/home/m/mehrad/brikiyou/scratch/slurm-%j.err

#SBATCH --gpus-per-node=0


module load BalamEnv

source /home/m/mehrad/brikiyou/scratch/new_spock_venv/bin/activate

if [[ "gpt-4o" == "llama" ]]; then
    source /home/m/mehrad/brikiyou/scratch/to_run.sh
    ollama serve > /home/m/mehrad/brikiyou/scratch/ollama.log 2>&1 &
fi


python3 /home/m/mehrad/brikiyou/scratch/spock/slack_bot/scripts/process_pdf.py     --model "gpt-4o" \
    --paper "/home/m/mehrad/brikiyou/scratch/spock/slack_bot/papers/Data-science-based_reconstruction_of_3-D_membrane_pore_structure_using_a_single_2-D_micrograph.pdf" \
    --questions "" \
    --user_id "U073D78M8UT" \
    --channel_id "D073ZCA5L6N"
