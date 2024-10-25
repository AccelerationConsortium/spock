#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:0


module load BalamEnv
cd /home/m/mehrad/brikiyou/scratch/spock_package
source bin/activate
cd spock/scholarly_creative_work

# Directory containing PDF files
pdf_dir="/home/m/mehrad/brikiyou/scratch/spock_package/spock/scholarly_creative_work/papers/papers"

# Get the list of PDF files
pdf_files=("$pdf_dir"/*.pdf)

# Loop over each PDF file
for pdf_path in "${pdf_files[@]}"
do
    # Extract the PDF filename (e.g., 'document.pdf')
    pdf_name=$(basename "$pdf_path")
    
    # Control the number of jobs in the queue
    number_of_jobs=$(squeue -u mehrad -h -t pending,running -r | wc -l)
    while (( number_of_jobs > 1900 ))
    do
        sleep 20s
        number_of_jobs=$(squeue -u mehrad -h -t pending,running -r | wc -l)
    done

    # Submit the job, exporting the PDF name
    sbatch --export=pdf_name="$pdf_name" --output=logs/output_%j.txt --nice=12000 executable.sh
done

