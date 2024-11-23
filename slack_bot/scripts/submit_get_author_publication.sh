#!/bin/bash

AUTHOR=$1
COUNT=$2
USER_ID=$3
CHANNEL_ID=$4

# Create a temporary job script
JOB_SCRIPT=$(mktemp)

# Generate the Slurm job script dynamically based on the model
cat <<EOT > $JOB_SCRIPT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=0


module load BalamEnv
module load python/3.8
source /home/m/mehrad/brikiyou/scratch/new_spock_venv/bin/activate

cd to/path/slack_bot

python3 /path/to/spock_processor.py \
    --author "$AUTHOR" \
    --count "$COUNT" \
    --user_id "$USER_ID" \
    --channel_id "$CHANNEL_ID"
EOT

# Submit the job script
sbatch $JOB_SCRIPT
