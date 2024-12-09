#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/m/mehrad/brikiyou/scratch/slurm-%j.out
#SBATCH --error=/home/m/mehrad/brikiyou/scratch/slurm-%j.err

#SBATCH --gpus-per-node=0


module load BalamEnv

source /home/m/mehrad/brikiyou/scratch/new_spock_venv/bin/activate

if [[ "gpt-4o" == "llama3.3" ]]; then
    source /home/m/mehrad/brikiyou/scratch/to_run.sh
    ollama serve > /home/m/mehrad/brikiyou/scratch/ollama.log 2>&1 &
fi


python3 /home/m/mehrad/brikiyou/scratch/spock/slack_bot/scripts/process_pdf.py     --model "gpt-4o" \
    --paper "/home/m/mehrad/brikiyou/scratch/spock/slack_bot/papers/Correction_to_Up-regulation_of_autophagy_is_a_mechanism_of_resistance_to_chemotherapy_and_can_be_inhibited_by_pantoprazole_to_increase_drug_sensitivity.pdf" \
    --questions "are llms mentioned here || what about material science ?" \
    --user_id "U073D78M8UT" \
    --channel_id "D073ZCA5L6N"
