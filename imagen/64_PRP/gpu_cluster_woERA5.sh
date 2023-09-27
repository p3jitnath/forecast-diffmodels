#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpgpuB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pritthijit.nath.ml@gmail.com
/vol/bitbucket/pn222/venv/bin/python /homes/pn222/Work/MSc_Project_pn222/imagen/64_PRP/m01w_woERA5_64_PRP.py -mode execute -epochs 261
