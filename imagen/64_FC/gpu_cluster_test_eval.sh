#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpgpuB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pritthijit.nath.ml@gmail.com

/vol/bitbucket/pn222/venv/bin/python /homes/pn222/Work/MSc_Project_pn222/imagen/64_FC/t07-test-set-evaluation.py -run_name 64_FC_rot904_sep_3e-4
/vol/bitbucket/pn222/venv/bin/python /homes/pn222/Work/MSc_Project_pn222/imagen/64_FC/t07-test-set-evaluation.py -run_name 64_FC_rot904_3e-4
/vol/bitbucket/pn222/venv/bin/python /homes/pn222/Work/MSc_Project_pn222/imagen/64_FC/t07-test-set-evaluation.py -run_name 64_FC_3e-4
