#!/bin/bash

# Group
#SBATCH --partition=GPU 

# Name
#SBATCH --job-name=BPMulT

# Tasks per node
#SBATCH --ntasks-per-node=24

# Tasks
#SBATCH --ntasks=24

# Log screen
#SBATCH --output=/home/Users/[BPMulT_examplePath]/outputs/out.log

# Max memory
#SBATCH --mem=0

# Max time
#SBATCH --time=0

# Task route path
Tasks=/home/Users/[BPMulT_examplePath]/jobs_to_run.txt
mpirun.openmpi -np ${SLURM_NTASKS} /opt/ClusterTools/MPI_Scheduler $Tasks 1


# to run: sbatch name_of_this_file
