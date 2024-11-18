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
$(if [[ "$MODEL" == "llama" ]]; then echo "#SBATCH --gpus-per-node=4"; echo "#SBATCH -p compute_full_node"; else echo "#SBATCH --gpus-per-node=0"; fi)


module load BalamEnv
module load python/3.8
source /home/m/mehrad/brikiyou/scratch/new_spock_venv/bin/activate

source 
if [[ "$MODEL" == "llama" ]]; then
    source /home/m/mehrad/brikiyou/scratch/to_run.sh
    ollama serve > ollama.log 2>&1 &
fi

python3 /path/to/spock_processor.py \
    --model "$MODEL" \
    --paper "$DOI_OR_TITLE" \
    --questions "$QUESTIONS_STR" \
    --user_id "$USER_ID" \
    --channel_id "$CHANNEL_ID"
EOT

# Submit the job script
sbatch $JOB_SCRIPT
