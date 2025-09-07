#!/bin/bash

inputfile=${1}
fold=${2}
i=${3}

mkdir -p $HOME/NAMD/msgs/
cd $HOME/NAMD/msgs/

echo "#!/bin/bash
#SBATCH --nodes=1
#SBATCH --constraint=cpu
##SBATCH -q debug
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --mem=1GB
#SBATCH -t 24:00:00

#SBATCH -J NAMD
#SBATCH --mail-user=user@domain.com
#SBATCH --mail-type=END,FAIL
module load conda

conda activate pyenv

mkdir -p $SCRATCH/NAMD/
cd $SCRATCH/NAMD/

# echo \"test\"

srun python3 $HOME/NAMD/serial.py ${inputfile} ${fold} ${i}

" > run.sl

#send interrupt signal 90 seconds before job is terminated
#sbatch --signal=INT@90 run.sl

sbatch run.sl

rm run.sl
