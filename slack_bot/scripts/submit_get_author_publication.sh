#!/bin/bash

AUTHOR=$1
COUNT=$2
USER_ID=$3
CHANNEL_ID=$4
JOBSCRIPT_PATH=$5

JOB_SCRIPT=$JOBSCRIPT_PATH


cat <<EOT > $JOB_SCRIPT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:02:00
#SBATCH --gpus-per-node=0


module load BalamEnv
module load python/3.8
source /home/m/mehrad/brikiyou/scratch/new_spock_venv/bin/activate


python3 /home/m/mehrad/brikiyou/scratch/spock/slack_bot/scripts/process_get_author_publication.py \
    --author "$AUTHOR" \\
    --count "$COUNT" \\
    --user_id "$USER_ID" \\
    --channel_id "$CHANNEL_ID"
EOT

# Submit the job script
tmux new-session -d -s temp_session "ssh -4 balam-login01 'sbatch $JOB_SCRIPT'"