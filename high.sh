#!/bin/bash
#SBATCH --job-name=pitch_hi
#SBATCH -n 6
#SBATCH --mem 50000
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH -p high                     # Partition to submit to
#SBATCH -o %x-%j.out # File to which STDOUT will be written
#SBATCH -e %x-%j.err # File to which STDERR will be written


module load torchaudio
module load SoundFile
module load PyTorch
module load CUDA
module load pandas
module load librosa
module load scikit-learn

echo "modules loaded"
python ~/instrument_pitch_tracker/windowed/train.py

