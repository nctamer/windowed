
import csv
import numpy as np
import librosa
import os
from torch.utils import data
from mir_eval import melody
from scipy.stats import norm
from six.moves import cPickle as pickle


AUDIO_SR = 16000
WINDOW_LEN = 1024

LABEL = {
    "n_bins": 360,
    "min_f0_hz": 31.70,
    "granularity_c": 20,
    "smooth_std_c": 25
}


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


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


class PrepareDataset(data.Dataset):
    def __init__(self, audio_folder, annotation_folder, save_folder, audio_file_extension="wav", save_size=512):
        super().__init__()
        self.annotations = [(_[:-4],  # name
                             os.path.join(audio_folder, _[:-4] + "." + audio_file_extension),  # audio file
                             csv.DictReader(open(os.path.join(annotation_folder, _)), fieldnames=['time', 'f0']))
                            for _ in sorted(os.listdir(annotation_folder)) if _.endswith(".csv")]
        self.label = Label(**LABEL)
        self.save_size = save_size
        self.save_folder = save_folder

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        name, filename, annotation = self.annotations[index]
        audio = librosa.load(filename, sr=AUDIO_SR, mono=True)[0]

        segments, labels, f0s, times = [], [], [], []
        i = 0
        for r, row in enumerate(annotation):
            time, f0 = int(float(row['time'])*AUDIO_SR), float(row['f0'])
            start, end = time - (WINDOW_LEN//2), time + (WINDOW_LEN//2)
            if (f0 > 0) & (start >= 0) & (end < len(audio)):
                segments.append(audio[start:end])
                labels.append(self.label.hz2label(f0))
                f0s.append(f0)
                times.append(time)
                i += 1
                if not i % self.save_size:
                    save_path = os.path.join(self.save_folder, name + "_" + str((i // self.save_size)-1))
                    save_dict({"times": np.array(times),
                               "f0s": np.array(f0s),
                               "segments": np.array(segments),
                               "labels": np.array(labels)}, save_path)
                    segments, labels, f0s, times = [], [], [], []
        if i % self.save_size:  # for the remainder
            save_path = os.path.join(self.save_folder, name + "_" + str(i // self.save_size))
            save_dict({"times": np.array(times),
                       "f0s": np.array(f0s),
                       "segments": np.array(segments),
                       "labels": np.array(labels)}, save_path)
        return index


if __name__ == '__main__':

    data_path = "/homedtic/ntamer/instrument_pitch_tracker/data/MDB-stem-synth"

    dataset = PrepareDataset(audio_folder=os.path.join(data_path, "audio_stems"),
                             annotation_folder=os.path.join(data_path, "annotation_stems"),
                             save_folder=os.path.join(data_path, "prep"), save_size=512)
    for n in dataset:
        print(n)
