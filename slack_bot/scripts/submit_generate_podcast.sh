#!/bin/bash

PAPER=$1
USER_ID=$2
CHANNEL_ID=$3
INITIAL_COMMENT=$4

# Define the path for the job script file
JOB_SCRIPT="/home/m/mehrad/brikiyou/scratch/spock/slack_bot/generated_job_script.sh"

# Generate the SLURM job script dynamically
cat <<EOT > $JOB_SCRIPT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:02:00
#SBATCH --gpus-per-node=0
#SBATCH --output=/home/m/mehrad/brikiyou/scratch/slurm-%j.out
#SBATCH --error=/home/m/mehrad/brikiyou/scratch/slurm-%j.err

module load BalamEnv
module load python/3.8
source /home/m/mehrad/brikiyou/scratch/new_spock_venv/bin/activate

cd /home/m/mehrad/brikiyou/scratch/spock/slack_bot

python3 /home/m/mehrad/brikiyou/scratch/spock/slack_bot/scripts/process_generate_podcast.py \\
    --paper "$PAPER" \\
    --user_id "$USER_ID" \\
    --channel_id "$CHANNEL_ID" \\
    --initial_comment "$INITIAL_COMMENT"
EOT

tmux new-session -d -s temp_session "ssh -4 balam-login01 'sbatch $JOB_SCRIPT'"