# LSBATCH: User input
#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J TotalSegmentationDice
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot --
#BSUB -R "rusage[mem=64GB]"
### -- set walltime limit: hh:mm --
#BSUB -W 24:00
# -- Notify me by email when execution begins --
#BSUB -B
# -- Notify me by email when execution ends   --
#BSUB -N
# -- email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s204211@dtu.dk
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o logs/Output_%J.out
#BSUB -e logs/Output_%J.err

# -- commands you want to execute --

source /zhome/c8/0/156532/anaconda3/bin/activate && conda activate ThesisEnv && python TrainSkipReduction.py

