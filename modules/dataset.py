
import csv
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
from torch_audiomentations import Compose, Gain, PolarityInversion, LowPassFilter, AddBackgroundNoise


AUDIO_SR = 16000
WINDOW_LEN = 1024

LABEL = {
    "n_bins": 360,
    "min_f0_hz": 31.70,
    "granularity_c": 20,
    "smooth_std_c": 25
}


class Label:
    def __init__(self, n_bins, min_f0_hz, granularity_c, smooth_std_c):
        self.n_bins = n_bins
        self.min_f0_hz = min_f0_hz
        self.min_f0_c = melody.hz2cents(np.array([min_f0_hz]))[0]
        self.granularity_c = granularity_c
        self.smooth_std_c = smooth_std_c
        self.pdf_normalizer = norm.pdf(0)
        self.centers_c = np.linspace(0, (self.n_bins - 1) * self.granularity_c, self.n_bins) + self.min_f0_c

    def c2label(self, pitch_c):
        """
        Converts pitch labels in cents, to a vector representing the classification label
        Uses the normal distribution centered at the pitch and the standard deviation of 25 cents,
        normalized so that the exact prediction has the value 1.0.
        :param pitch_c: a number or numpy array of shape (1,)
        pitch values in cents, as returned by hz2cents with base_frequency = 10 (default)
        :return: ndarray
        """
        result = norm.pdf((self.centers_c - pitch_c) / self.smooth_std_c).astype(np.float32)
        result /= self.pdf_normalizer
        return result

    def hz2label(self, pitch_hz):
        pitch_c = melody.hz2cents(np.array([pitch_hz]))[0]
        return self.c2label(pitch_c)


class DictDataset(data.Dataset):
    def __init__(self, prep_folder, instrument_name=None):
        super().__init__()
        if instrument_name:
            self.files, self.audio_names, ids = zip(*[[os.path.join(prep_folder, _)] + _.rsplit("_", 1)
                                                    for _ in sorted(os.listdir(prep_folder)) if instrument_name in _])
        else:
            self.files, self.audio_names, ids = zip(*[[os.path.join(prep_folder, _)] + _.rsplit("_", 1)
                                                    for _ in sorted(os.listdir(prep_folder))])
        self.audio_map = {_: [] for _ in set(self.audio_names)}
        for i_file, file in enumerate(self.files):
            audio_name = self.audio_names[i_file]
            self.audio_map[audio_name].append(file)
        self.audio_list = sorted(self.audio_map.keys())
        self.label = Label(**LABEL)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = load_dict(self.files[index])

        return self.audio_names[index], file['segments'], file['labels'], file['f0s']


class AudioDataset(data.Dataset):
    def __init__(self, audio_folder, annotation_folder, audio_file_extension="wav"):
        super().__init__()
        self.annotations = [(os.path.join(audio_folder, _[:-4] + "." + audio_file_extension),
                             csv.DictReader(open(os.path.join(annotation_folder, _)), fieldnames=['time', 'f0']))
                            for _ in sorted(os.listdir(annotation_folder)) if _.endswith(".csv")]
        self.label = Label(**LABEL)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        filename, annotation = self.annotations[index]
        audio = librosa.load(filename, sr=AUDIO_SR, mono=True)[0]

        segments, labels, f0s = [], [], []
        for row in annotation:
            time, f0 = int(float(row['time'])*AUDIO_SR), float(row['f0'])
            start, end = time - (WINDOW_LEN//2), time + (WINDOW_LEN//2)
            if (f0 > 0) & (start >= 0) & (end < len(audio)):
                segments.append(audio[start:end])
                labels.append(self.label.hz2label(f0))
                f0s.append(f0)

        return filename, np.array(segments), np.array(labels), np.array(f0s)


class Collator(object):
    def __init__(self, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, batch):
        name, segments, label, f0 = zip(*batch)
        segments = np.vstack(segments)
        label = np.vstack(label)
        f0 = np.hstack(f0)

        if self.shuffle:  # shuffle the data further
            shuffler = np.random.permutation(len(label))
            segments = segments[shuffler]
            label = label[shuffler]
            f0 = f0[shuffler]

        # remove the last batch smaller than the batch size and convert them into tensors
        n_batches = int(np.floor(len(label)/self.batch_size))*self.batch_size
        segments = torch.tensor(
            segments[:n_batches].reshape((n_batches//self.batch_size, self.batch_size, segments.shape[-1])),
            dtype=torch.float)
        label = torch.tensor(
            label[:n_batches].reshape((n_batches//self.batch_size, self.batch_size, label.shape[-1])),
            dtype=torch.float32)
        f0 = torch.tensor(
            f0[:n_batches].reshape((n_batches//self.batch_size, self.batch_size)), dtype=torch.float32)
        return segments, label, f0


def get_part(main_dataset, indices):
    part = copy.copy(main_dataset)
    if hasattr(main_dataset, 'audio_list'):
        audio_names = [main_dataset.audio_list[idx] for idx in indices]
        part.files = []
        for name in audio_names:
            part.files.extend(main_dataset.audio_map[name])
    else:
        part.annotations = [main_dataset.annotations[ind] for ind in indices]
    return part


def partition_dataset(main_dataset, dev_ratio=0.2, test_ratio=0.2):
    """
    Partitioning based on tracks
    A better version should definitely consider track durations
    """
    if hasattr(main_dataset, 'audio_list'):
        idx = set(range(main_dataset.audio_list.__len__()))
    else:
        idx = set(range(main_dataset.__len__()))
    dev_count = int(len(idx) * dev_ratio)
    test_count = int(len(idx) * test_ratio)
    train_count = len(idx) - dev_count - test_count

    train_idx = sorted(random.sample(idx, train_count))
    dev_idx = sorted(random.sample(list(set(idx).difference(set(train_idx))), dev_count))
    test_idx = sorted(list(set(idx) - set(train_idx) - set(dev_idx)))
    return get_part(main_dataset, train_idx), get_part(main_dataset, dev_idx), get_part(main_dataset, test_idx)


if __name__ == '__main__':

    data_path = "/home/nazif/PycharmProjects/data/Bach10-mf0-synth"
    device = 'cpu'

    # Initialize augmentation callable
    apply_augmentation = Compose(
        transforms=[
            LowPassFilter(
                p=0.5
            ),
            Gain(
                min_gain_in_db=-15.0,
                max_gain_in_db=5.0,
                p=0.5,
            ),
            PolarityInversion(p=0.5),
            AddBackgroundNoise(
                background_paths="../test_fixtures/bg_short", p=0.5
            ),
        ]
    )

    files_per_batch = 4  # the number of batches (separate files) we read in the loader
    batch_sample_size = 256  # the real batch size the GPU sees
    dataset = DictDataset(os.path.join(data_path, "prep"), instrument_name="violin")

    # train_set, dev_set, test_set = partition_dataset(dataset, dev_ratio=0.2, test_ratio=0.2)
    collate = Collator(batch_size=batch_sample_size, shuffle=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=files_per_batch, shuffle=True, collate_fn=collate)
    for (s, l, f) in loader:
        for i, sequence in enumerate(s):
            perturbed_audio_samples = apply_augmentation(sequence.unsqueeze(1), sample_rate=16000).squeeze(1)
            sequence = sequence.to(device)
            target = l[i].to(device)

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make an example tensor with white noise.
    # This tensor represents 8 audio snippets with 2 channels (stereo) and 2 s of 16 kHz audio.
    audio_samples = torch.rand(size=(8, 2, 32000), dtype=torch.float32, device=torch_device) - 0.5

    # Apply augmentation. This varies the gain and polarity of (some of)
    # the audio snippets in the batch independently.
    perturbed_audio_samples = apply_augmentation(audio_samples, sample_rate=16000)

