
import csv
import io
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


class Dataset(data.Dataset):
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
            dtype=torch.long)
        label = torch.tensor(
            label[:n_batches].reshape((n_batches//self.batch_size, self.batch_size, label.shape[-1])),
            dtype=torch.float32)
        f0 = torch.tensor(
            f0[:n_batches].reshape((n_batches//self.batch_size, self.batch_size)), dtype=torch.float32)
        return segments, label, f0


def get_part(main_dataset, indices):
    part = copy.copy(main_dataset)
    part.annotations = [main_dataset.annotations[ind] for ind in indices]
    return part


def partition_dataset(main_dataset, dev_ratio=0.2, test_ratio=0.2):
    """
    Partitioning based on tracks
    A better version should definitely consider track durations
    """
    idx = set(range(main_dataset.annotations.__len__()))
    dev_count = int(len(main_dataset) * dev_ratio)
    test_count = int(len(main_dataset) * test_ratio)
    train_count = len(main_dataset) - dev_count - test_count

    train_idx = sorted(random.sample(idx, train_count))
    dev_idx = sorted(random.sample(list(set(idx).difference(set(train_idx))), dev_count))
    test_idx = sorted(list(set(idx) - set(train_idx) - set(dev_idx)))
    return get_part(main_dataset, train_idx), get_part(main_dataset, dev_idx), get_part(main_dataset, test_idx)


if __name__ == '__main__':

    data_path = "/home/nazif/PycharmProjects/data/MDB-stem-synth"
    device = 'cpu'

    files_per_batch = 4  # the number of batches (separate files) we read in the loader
    batch_sample_size = 128  # the real batch size the GPU sees
    dataset = Dataset(os.path.join(data_path, "audio_stems"), os.path.join(data_path, "annotation_stems"))
    collate = Collator(batch_size=batch_sample_size, shuffle=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=files_per_batch, shuffle=True, collate_fn=collate)
    for (s, l, f) in loader:
        for i, sequence in enumerate(s):
            sequence = sequence.to(device)
            target = l[i].to(device)
