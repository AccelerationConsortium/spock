#!/bin/bash

MODEL=$1
DOI_OR_TITLE=$2
QUESTIONS_STR=$3
USER_ID=$4
CHANNEL_ID=$5

# Create a temporary job script
JOB_SCRIPT=$(mktemp)

# Generate the Slurm job script dynamically based on the model
cat <<EOT > $JOB_SCRIPT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
$(if [[ "$MODEL" == "llama" ]]; then echo "#SBATCH --gpus-per-node=4"; echo "#SBATCH -p compute_full_node"; else echo "#SBATCH --gpus-per-node=1"; fi)


module load BalamEnv
source /home/m/mehrad/brikiyou/scratch/new_spock_venv/bin/activate

if [[ "$MODEL" == "llama" ]]; then
    source /home/m/mehrad/brikiyou/scratch/to_run.sh
    ollama serve > ollama.log 2>&1 &
fi

python3 /home/m/mehrad/brikiyou/scratch/spock/slack_bot/scripts/process_publication.py \
    --model "$MODEL" \\
    --paper "$DOI_OR_TITLE" \\
    --questions "$QUESTIONS_STR" \\
    --user_id "$USER_ID" \\
    --channel_id "$CHANNEL_ID"
EOT
cd 
cd scratch/

# Submit the job script
sbatch $JOB_SCRIPT
