#!/bin/bash
#SBATCH --job-name=crepe_med
#SBATCH -n 4
#SBATCH --time=8:00:00
#SBATCH --mem 12000
#SBATCH --partition=medium
# #SBATCH --gres=gpu:1
#SBATCH -p medium                     # Partition to submit to
#SBATCH -o %x-%j.out # File to which STDOUT will be written
#SBATCH -e %x-%j.err # File to which STDERR will be written

echo "now start" 

module load torchaudio
module load PyTorch
# module load CUDA
module load pandas
module load librosa
module load scikit-learn
python ~/instrument_pitch_tracker/windowed/train.py > ~/instrument_pitch_tracker/windowed/out.txt

cudaMemTest=/soft/slurm_templates/bin/cuda_memtest-1.2.3/cuda_memtest
cudaDevs=$(echo $CUDA_VISIBLE_DEVICES | sed -e 's/,/ /g')
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES

