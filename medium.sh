#!/bin/bash
#SBATCH --job-name=crepe_med
#SBATCH -n 17
#SBATCH --time=8:00:00
#SBATCH --mem 64000
#SBATCH --partition=medium
#SBATCH --gres=gpu:1
#SBATCH -p medium                     # Partition to submit to
#SBATCH -o %x-%j.out # File to which STDOUT will be written
#SBATCH -e %x-%j.err # File to which STDERR will be written

# I. Define directory names [DO NOT CHANGE]
# =========================================

# get name of the temporary directory working directory, physically on the compute-node
workdir="${TMPDIR}"

# get submit directory
# (every file/folder below this directory is copied to the compute node)
submitdir="/homedtic/ntamer/instrument_pitch_tracker/data/"

# 1. Transfer to node [DO NOT CHANGE]
# ===================================

# create/empty the temporary directory on the compute node
if [ ! -d "${workdir}" ]; then
  mkdir -p "${workdir}"
else
  rm -rf "${workdir}"/*
fi

# change current directory to the location of the sbatch command
# ("submitdir" is somewhere in the home directory on the head node)
cd "${submitdir}"
# copy all files/folders in "submitdir" to "workdir"
# ("workdir" == temporary directory on the compute node)
cp -prf * ${workdir}
# change directory to the temporary directory on the compute-node
cd ${workdir}

echo "now start" 

module load torchaudio
module load PyTorch
module load CUDA
module load pandas
module load librosa
module load scikit-learn
python ~/instrument_pitch_tracker/windowed/train.py > ~/instrument_pitch_tracker/windowed/out.txt

cudaMemTest=/soft/slurm_templates/bin/cuda_memtest-1.2.3/cuda_memtest
cudaDevs=$(echo $CUDA_VISIBLE_DEVICES | sed -e 's/,/ /g')
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES

