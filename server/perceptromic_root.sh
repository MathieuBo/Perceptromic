#!/usr/bin/env bash

#############################
# Your job name (displayed by the queue)
#PBS -N Perceptromic0

#change output file's name
#PBS -e /home/anioche/mathieu/.log/Perceptromic0.err
#PBS -o /home/anioche/mathieu/.log/Perceptromic0.log

# Specify the working directory
#PBS -d /home/anioche/mathieu/perceptron_avakas

# walltime (hh:mm::ss)
#PBS -l walltime=04:00:00

# Specify the number of nodes(nodes=) and the number of cores per nodes(ppn=) to be used
#PBS -l nodes=1:ppn=6

# Specify physical memory: kb for kilobytes, mb for megabytes, gb for gigabytes
#PBS -l mem=300mb

#PBS -m abe
#PBS -M mbourdenx@me.com

# fin des directives PBS
#############################

module purge # modules cleaning
module add torque
pyenv local 3.5.2
# module add python3
# module add gcc/4.8.2 # For cpp

# useful informations to print
echo "#############################"
echo "User:" ${USER}
echo "Date:" `date`
echo "Host:" `hostname`
echo "Directory:" `pwd`
echo "PBS_JOBID:" ${PBS_JOBID}
echo "PBS_O_WORKDIR:" ${PBS_O_WORKDIR}
echo "PBS_NODEFILE: " `cat ${PBS_NODEFILE} | uniq`
echo "#############################"

#############################

# What you actually want to launch
echo "Start the job"
# launch python script with pickle object for parameters and number of processes
python launch_multi.py perceptromic_job0.p 12 0

# all done
echo "Job finished"
