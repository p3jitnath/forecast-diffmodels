#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpgpuB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pritthijit.nath.ml@gmail.com
/vol/bitbucket/pn222/venv/bin/python /homes/pn222/Work/MSc_Project_pn222/imagen/64_128/tn2-sampling-and-evaluation.py -run_name 64_128_rot904_sep_3e-4
