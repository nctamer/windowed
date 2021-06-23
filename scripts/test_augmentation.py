import csv
from scipy.io.wavfile import write
import numpy as np
import librosa
import librosa.display
import os
from torch.utils import data
import torch
from mir_eval import melody
from scipy.stats import norm
import random
import copy
from scripts.prep_data import load_dict
from torch_audiomentations import *


if __name__ == '__main__':

    data_path = "/home/nazif/PycharmProjects/data/Bach10-mf0-synth"
    save_path = "/home/nazif/PycharmProjects/data/audiomentation_samples"
    device = 'cpu'

    # Initialize augmentation callable
    apply_augmentation = Compose(
        transforms=[
            Gain(
                min_gain_in_db=-15.0,
                max_gain_in_db=5.0,
                p=1.0, mode="per_batch"
            ),
            PolarityInversion(p=1.0),
            ApplyImpulseResponse(ir_paths="../test_fixtures/ir", p=1.0),
            AddColoredNoise(min_snr_in_db=10.0,
                            max_snr_in_db=30.0,
                            p=1.0, mode="per_batch"),
            AddBackgroundNoise(
                background_paths="../test_fixtures/bg_short",
                min_snr_in_db=5.0,
                max_snr_in_db=10.0,
                p=1.0,
                mode="per_batch"
            ),
        ]
    )

    os.makedirs(save_path, exist_ok=True)  # the real batch size the GPU sees
    file = os.path.join(data_path, "audio_stems")
    file = os.path.join(file, "03_ChristederdubistTagundLicht_violin.RESYN.wav")
    audio = librosa.load(file, sr=16000, mono=True)[0]
    write(os.path.join(save_path, "original.wav"), 16000, audio.astype(np.float32))
    for aug in apply_augmentation.transforms:
        perturbed_audio = aug(torch.Tensor(audio).view(1,1,-1), sample_rate=16000).view(-1).numpy()
        write(os.path.join(save_path, aug._get_name() + ".wav"), 16000, perturbed_audio.astype(np.float32))
    combined_audio = apply_augmentation(torch.Tensor(audio).view(1,1,-1), sample_rate=16000).view(-1).numpy()
    write(os.path.join(save_path, "combined.wav"), 16000, combined_audio.astype(np.float32))

