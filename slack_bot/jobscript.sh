#!/bin/bash
#SBATCH --nodes=1
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:0

source /home/m/mehrad/brikiyou/scratch/to_run.sh


ollama serve > ollama.log 2>&1 &

cd /home/m/mehrad/brikiyou/scratch/spock_package
source bin/activate
# create a new terminal and run ollama serve
cd spock/scholarly_creative_work

python3 run.py "$name"  # To change

rm -rf slurm*
