#!/bin/bash

DOI_OR_TITLE=$1
QUESTIONS_STR=$2
USER_ID=$3
CHANNEL_ID=$4
JOBSCRIPT_PATH=$5

JOB_SCRIPT=$JOBSCRIPT_PATH

cat <<EOT > $JOB_SCRIPT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:05:00
$(if [[ "$MODEL" == "llama3.3" ]]; then echo "#SBATCH --gpus-per-node=4"; echo "#SBATCH -p compute_full_node"; else echo "#SBATCH --gpus-per-node=1"; fi)
#SBATCH --output=/home/m/mehrad/brikiyou/scratch/slurm-%j.out
#SBATCH --error=/home/m/mehrad/brikiyou/scratch/slurm-%j.err


module load BalamEnv

source /home/m/mehrad/brikiyou/scratch/new_spock_venv/bin/activate

if [[ "$MODEL" == "llama3.3" ]]; then
    source /home/m/mehrad/brikiyou/scratch/to_run.sh
    ollama serve > /home/m/mehrad/brikiyou/scratch/ollama.log 2>&1 &
fi

python3 /home/m/mehrad/brikiyou/scratch/spock/slack_bot/scripts/process_publication.py \
    --publication "$DOI_OR_TITLE" \\
    --questions "$QUESTIONS_STR" \\
    --user_id "$USER_ID" \\
    --channel_id "$CHANNEL_ID"
EOT

tmux new-session -d -s temp_session "ssh -4 balam-login01 'sbatch $JOB_SCRIPT'"