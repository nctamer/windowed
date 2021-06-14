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


module load torchaudio
module load PyTorch
# module load CUDA
module load pandas
module load librosa
module load scikit-learn

echo "modules loaded"
python ~/instrument_pitch_tracker/windowed/train.py > ~/instrument_pitch_tracker/windowed/out.txt

