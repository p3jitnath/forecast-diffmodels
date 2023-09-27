#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpgpuB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pritthijit.nath.ml@gmail.com

/vol/bitbucket/pn222/venv/bin/python /homes/pn222/Work/MSc_Project_pn222/imagen/64_FC/tw5-forecasting-pipeline.py -region "North Indian Ocean" -name "Amphan" -horizon 100 -start 0
/vol/bitbucket/pn222/venv/bin/python /homes/pn222/Work/MSc_Project_pn222/imagen/64_FC/tw5-forecasting-pipeline.py -region "North Indian Ocean" -name "Mocha" -horizon 100 -start 0
/vol/bitbucket/pn222/venv/bin/python /homes/pn222/Work/MSc_Project_pn222/imagen/64_FC/tw5-forecasting-pipeline.py -region "North Indian Ocean" -name "Tauktae" -horizon 100 -start 0
/vol/bitbucket/pn222/venv/bin/python /homes/pn222/Work/MSc_Project_pn222/imagen/64_FC/tw5-forecasting-pipeline.py -region "North Indian Ocean" -name "Maha" -horizon 100 -start 24