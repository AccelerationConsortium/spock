#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=0:30:00
#SBATCH --output=/home/m/mehrad/brikiyou/scratch/spock/examples/nsys_benchmarking/out/slurm-%j.out
#SBATCH --error=/home/m/mehrad/brikiyou/scratch/spock/examples/nsys_benchmarking/out/slurm-%j.err
#SBATCH --mail-user=youssef.briki@umontreal.ca
#SBATCH --mail-type=FAIL 
#SBATCH -p debug_full_node

module load BalamEnv
source /home/m/mehrad/brikiyou/scratch/to_run.sh
source /home/m/mehrad/brikiyou/scratch/nsys_setup.sh
ollama serve > /home/m/mehrad/brikiyou/scratch/ollama.log 2>&1 &
cd /home/m/mehrad/brikiyou/scratch/spock/examples/
nsys profile --trace=cuda,nvtx --gpu-metrics-devices=all --output=ollama_cli_4gpu --force-overwrite true ollama run llama3.3 "why is the sky blue?"
