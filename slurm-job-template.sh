#!/bin/bash

# Submit this job with "sbatch <filename>"

#SBATCH --clusters=<cluster>   # Cluster name
#SBATCH --job-name="My job"    # Set job name
#SBATCH --chdir=/path/to/wd    # Set working directory
#SBATCH --time=6-0             # Time limit
#SBATCH --output=slurm-%x.out  # stdout and stderr

# This will run the script for month 3 (March) and run 0.
srun -N1 bash -c "source venv/bin/activate; cd source; python3.9 generate-points.py 3 0; deactivate" &

# Add more to a single job (add --nodes option above accordingly) or submit multiple jobs
# srun -N1 bash -c "source venv/bin/activate; cd source; python3.9 generate-points.py 3 1; deactivate" &

wait # wait for all jobs in this script to complete
