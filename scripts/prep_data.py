from modules.dataset import Label
import csv
import numpy as np
import librosa
import os
from torch.utils import data
from mir_eval import melody
from scipy.stats import norm
from six.moves import cPickle as pickle


AUDIO_SR = 44100
WINDOW_LEN = 2048

LABEL = {
    "n_bins": 720,
    "min_f0_hz": 32.7032,
    "granularity_c": 10,
    "smooth_std_c": 12,
}


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di




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

    """ Content of prep44100 folder:
    - SR: 44100
    - window length: 1024
    - #segments per file: 256"""

    """ Content of prep2048 folder:
    - SR: 44100
    - window length: 2048
    - #segments per file: 256
    - LABEL "n_bins": 720, "min_f0_hz": 32.7032, "granularity_c": 10, "smooth_std_c": 12
    """

    base_path = "/homedtic/ntamer/instrument_pitch_tracker/data"
    #data_path = "/home/nazif/PycharmProjects/data/Bach10-mf0-synth"

    for data_path in ["Bach10-mf0-synth", "MDB-stem-synth"]:
        data_path = os.path.join(base_path, data_path)

        dataset = PrepareDataset(audio_folder=os.path.join(data_path, "audio_stems"),
                                 annotation_folder=os.path.join(data_path, "annotation_stems"),
                                 save_folder=os.path.join(data_path, "prep2048"), save_size=256)
        for n in dataset:
            print(n)
