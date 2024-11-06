#!/bin/bash
# Sample slurm submission script for the Chimera
# compute cluster
# Lines beginning with # are comments, and will be ignored by
# the interpreter.  Lines beginning with #SBATCH are
# directives to the scheduler.  These in turn can be
# commented out by adding a second # (e.g. ##SBATCH lines
# will not be processed by the scheduler).
#
#
# set name of job
#SBATCH --job-name=EXP5-10000images-3epochs
#
# set the number of processors/tasks needed
##SBATCH -n 4
# for hyperthreaded,shared memory jobs, set 1 task, 1 node,
# and set --cpus-per-task to total number of threads

#SBATCH -n 1
#SBATCH --cpus-per-task=4

# set the number of Nodes needed.  Set to 1 for shared 
# memory jobs
#SBATCH -N 1

#set an account to use
#if not used then default will be used
# for scavenger users, use this format:
##SBATCH --account=pi_first.last
# for contributing users, use this format:
##SBATCH --account=<deptname|lastname>

# set max wallclock time  DD-HH:MM:SS
#SBATCH --time=00-30:00:00

# set a memory request
#SBATCH --mem=128gb

# Set filenames for stdout and stderr.  %j can be used
# for the jobid.
# see "filename patterns" section of the sbatch man page for
# additional options
#SBATCH --error=%x-%j.err
#SBATCH --output=%x-%j.out


# set the partition where the job will run.  Multiple partitions can
# be specified as a comma separated list
# Use command "sinfo" to get the list of partitions
##SBATCH --partition=Intel6240
#SBATCH --partition=DGXA100
# restricting inheritance of environment variables is
# required for chimera12 and 13:
# if this option is used, source /etc/profile below.
#SBATCH --export=HOME

#Optional
# mail alert at start, end and/or failure of execution
# see the sbatch man page for other options
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=huuthanhvy.nguyen001@umb.edu

# Put your job commands here, including loading any needed
# modules or diagnostic echos.

# this job simply reports the hostname and sleeps for
# two minutes

# source the local profile.  This is recommended in
# conjunction with the --export=HOME or --export=NONE
# sbatch options.
#SBATCH --gres=gpu:A100:1
#SBATCH --requeue                   # Enable requeueing on preemption

. /etc/profile

. ~/.bashrc

echo "using $SLURM_CPUS_ON_NODE CPUs"
echo `date`

conda activate sbatch2

python EXP5Weber.py

# Diagnostic/Logging Information
echo "Finish Run"
echo "end time is `date`"

