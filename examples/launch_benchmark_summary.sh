#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5:00:00
#SBATCH --output=/home/m/mehrad/brikiyou/scratch/spock/slack_bot/out/slurm-%j.out
#SBATCH --error=/home/m/mehrad/brikiyou/scratch/spock/slack_bot/out/slurm-%j.err
#SBATCH --mail-user=youssef.briki@umontreal.ca
#SBATCH --mail-type=FAIL 


module load BalamEnv
module load python
source /home/m/mehrad/brikiyou/scratch/new_spock_venv_2/bin/activate
source /home/m/mehrad/brikiyou/scratch/to_run.sh
ollama serve > /home/m/mehrad/brikiyou/scratch/ollama.log 2>&1 &

python3 /home/m/mehrad/brikiyou/scratch/spock/examples/benchmark_llama3.1_sum.py