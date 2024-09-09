#!/bin/bash

MODEL=$1
PAPER_PATH=$2
QUESTIONS_STR=$3
USER_ID=$4
CHANNEL_ID=$5

# Create a temporary job script
JOB_SCRIPT=$(mktemp)

# Write the job script content
cat <<EOT > $JOB_SCRIPT
#!/bin/bash
#SBATCH --job-name=spock_job
#SBATCH --output=/path/to/logs/%j.out  # Replace with your log directory
#SBATCH --error=/path/to/logs/%j.err   # Replace with your error log directory

# Load necessary modules or activate virtual environment
# module load python/3.8
# source /path/to/venv/bin/activate

# Export necessary environment variables
export BOT_TOKEN='your-slack-bot-token'  # Ensure your bot token is securely managed

# Run the Python script
python3 /path/to/spock_processor.py \\
    --model "$MODEL" \\
    --paper "$PAPER_PATH" \\
    --questions "$QUESTIONS_STR" \\
    --user_id "$USER_ID" \\
    --channel_id "$CHANNEL_ID"
EOT

# Submit the job script
sbatch $JOB_SCRIPT

# Clean up the temporary job script
rm $JOB_SCRIPT
