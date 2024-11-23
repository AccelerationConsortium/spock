#!/bin/bash

PAPER=$1
AUDIO_FILE_PATH=$2
USER_ID=$3
CHANNEL_ID=$4
INITIAL_COMMENT=$5

# Create a temporary job script
JOB_SCRIPT=$(mktemp)

# Generate the Slurm job script dynamically based on the model
cat <<EOT > $JOB_SCRIPT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:01:00
#SBATCH --gpus-per-node=0


module load BalamEnv
module load python/3.8
source /home/m/mehrad/brikiyou/scratch/new_spock_venv/bin/activate

cd to/path/slack_bot

python3 /path/to/spock_processor.py \
    --paper "$PAPER" \
    --audio_file_path "$AUDIO_FILE_PATH" \
    --user_id "$USER_ID" \
    --channel_id "$CHANNEL_ID"
    --initial_comment "$INITIAL_COMMENT"
EOT

# Submit the job script
sbatch $JOB_SCRIPT
