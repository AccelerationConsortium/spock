#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:0
####################################################
##########MACHINE SPECIFIC DETAILS GO HERE##########
####################################################

module load BalamEnv
cd /home/m/mehrad/brikiyou/scratch/
source to_run.sh
cd /home/m/mehrad/brikiyou/scratch/spock_package
source bin/activate
cd spock/scholarly_creative_work

python3 run.py "$name"

rm -rf slurm*
