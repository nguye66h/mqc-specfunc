# mqc-specfunc

This code is based on https://github.com/mandalgrouptamu/SemiClassical-NAMD/
The submit.sh file is modified to submit jobs to SLURM with the assumption of a Python conda environment called pyenv. The user can fix the directories where output files are returned in submit.sh
Run by doing the following steps:
- check file in the Model folder to fix the approriate parameters
- go into submit.sh to adjust memory and time necessary
- go into run.py to check files to copy down approriately
- adjust parameters in input.txt
- run command:
python run.py input.txt {name of the output folder}

The "holstein" model can be run with the "mfe" and "mash" methods. The "lif" model can only be run with "mfe_kspace" method. The "mfe" and "mash" methods return amplitudes of the wavefunction every nuclear timestep while "mfe_kspace" method return the correlation function C_k(t) every electronic timestep.
