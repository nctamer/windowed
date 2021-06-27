import torch
import torch.nn as nn
import sys
import os
import numpy as np
import torchaudio
from dataset import Label, LABEL


class ConvBlock(nn.Module):
    def __init__(self, f, w, s, d, in_channels):
        super().__init__()
        p1 = d[0]*(w - 1) // 2
        p2 = d[0]*(w - 1) - p1
        self.pad = nn.ZeroPad2d((0, 0, p1, p2))

        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=f, kernel_size=(w, 1), stride=s, dilation=d)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(f)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class CREPE(nn.Module):
    def __init__(self, model_capacity="full"):
        super().__init__()

        capacity_multiplier = {
            'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32
        }[model_capacity]

        self.layers = [1, 2, 3, 4, 5, 6]
        filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
        filters = [1] + filters
        widths = [512, 64, 64, 64, 64, 64]
        strides = [(4, 1), (2, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
        dilation = [(1, 1), (3, 1), (5, 1), (7, 1), (7, 1), (7, 1)]

        for i in range(len(self.layers)):
            f, w, d, s, in_channel = filters[i + 1], widths[i], dilation[i], strides[i], filters[i]
            self.add_module("conv%d" % i, ConvBlock(f, w, s, d, in_channel))

        self.linear = nn.Linear(64 * capacity_multiplier, 720)
        # TODO: experiment with stride ib the second filter and the last linear into 128*capacityMul

    def load_weight(self, model_path):
        self.load_state_dict(torch.load(model_path, map_location=self.linear.weight.device))

    def forward(self, x):
        # x : shape (batch, sample)
        x = x.view(x.shape[0], 1, -1, 1)
        for i in range(len(self.layers)):
            x = self.__getattr__("conv%d" % i)(x)

        x = x.permute(0, 3, 2, 1)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x

    @staticmethod
    def get_frame(audio, hop_length=128):
        audio = nn.functional.pad(audio, pad=(1024, 1024))
        # make 1024-sample frames of the audio with hop length of 10 milliseconds
        n_frames = 1 + int((len(audio) - 2048) / hop_length)
        assert audio.dtype == torch.float32
        itemsize = 1  # float32 byte size
        frames = torch.as_strided(audio, size=(2048, n_frames), stride=(itemsize, hop_length * itemsize))
        frames = frames.transpose(0, 1).clone()

        frames -= (torch.mean(frames, axis=1).unsqueeze(-1))
        frames /= (torch.std(frames, axis=1).unsqueeze(-1))
        return frames

    def get_activation(self, audio, sr, hop_length=128, batch_size=128):
        """
        audio : (N,) or (C, N)
        """

        if sr != 44100:
            rs = torchaudio.transforms.Resample(sr, 44100)
            audio = rs(audio)

        if len(audio.shape) == 2:
            if audio.shape[0] == 1:
                audio = audio[0]
            else:
                audio = audio.mean(dim=0)  # make mono

        frames = self.get_frame(audio, hop_length)
        activation_stack = []
        device = self.linear.weight.device

        for i in range(0, len(frames), batch_size):
            f = frames[i:min(i + batch_size, len(frames))]
            f = f.to(device)
            act = self.forward(f)
            activation_stack.append(act.cpu())
        activation = torch.cat(activation_stack, dim=0)
        return activation

    def predict(self, audio, sr, label=Label(**LABEL), viterbi=False, hop_length=128, batch_size=128):
        activation = self.get_activation(audio, sr, batch_size=batch_size, hop_length=hop_length)
        frequency = label.salience2hz(activation, viterbi=viterbi)
        confidence = activation.max(dim=1)[0]
        time = torch.arange(confidence.shape[0]) * hop_length / 44100
        return time, frequency, confidence, activation

    def process_file(self, file, output=None, viterbi=False, hop_length=128, save_plot=False, batch_size=128,
                     label=Label(**LABEL)):
        try:
            audio, sr = torchaudio.load(file)
        except ValueError:
            print("CREPE-pytorch : Could not read", file, file=sys.stderr)

        with torch.no_grad():
            time, frequency, confidence, activation = self.predict(
                audio, sr,
                viterbi=viterbi,
                hop_length=hop_length,
                batch_size=batch_size,
                label=label
            )

        time, frequency, confidence, activation = time.numpy(), frequency.numpy(), confidence.numpy(), activation.numpy()
        data = np.vstack([time, frequency, confidence]).T
        if output:
            np.save(output + ".npy", data)
            np.savetxt(output + ".txt", data[:, :2], fmt=['%.6f', '%.6f'])

        # save the salience visualization in a PNG file
        if save_plot:
            import matplotlib.cm
            from imageio import imwrite

            plot_file = os.path.join(output, os.path.basename(os.path.splitext(file)[0])) + ".activation.png"
            # to draw the low pitches in the bottom
            salience = np.flip(activation, axis=1)
            inferno = matplotlib.cm.get_cmap('inferno')
            image = inferno(salience.transpose())

            imwrite(plot_file, (255 * image).astype(np.uint8))



if __name__ == "__main__":

    def get_audio_stem(name):
        return os.path.join(STEM_FOLDER, name + ".wav")

    DATA_FOLDER = "/home/nazif/PycharmProjects/resynthesis/data/"
    MODEL = "/home/nazif/PycharmProjects/models/dilated2048_Jun23_16/dilated2048"


    # cr = CREPE().cuda() #no cuda for debug
    cr = CREPE().cpu()
    cr.load_weight(MODEL)
    STEM_FOLDER = os.path.join(DATA_FOLDER, "original")
    #ANNO_FOLDER = DATA_FOLDER + "Bach10-mf0-synth/annotation_stems/"
    TARGET_FOLDER = os.path.join(DATA_FOLDER, "pitch_tracks")

    names = os.listdir(STEM_FOLDER)
    names = sorted([_[:-4] for _ in names if _.endswith(".wav")])

    from tqdm import tqdm

    for viterbi in (True, False):
        print("viterbi:", viterbi)
        for piece in tqdm(names):
            target_file = os.path.join(TARGET_FOLDER, piece) + ".dilated"
            if viterbi:
                target_file = target_file + "_viterbi"
            cr.process_file(get_audio_stem(piece), target_file, hop_length=128, viterbi=viterbi)
