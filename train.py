import pandas as pd
import numpy as np
import scipy.signal
from modules.crepe import CREPE, get_frame
from modules.utils import to_freq, eval_from_hz
from scipy import signal
import os
import sys
from torch.utils import data
import torch.nn as nn
from pathlib import Path
import torch
import torchaudio
import matplotlib.pyplot as plt
from modules.dataset import Collator, Dataset, partition_dataset
import pickle

LEARNING_RATE = 1e-6
MAX_EPOCH = 200
BATCH_SIZE = 512
BATCH_TRACKS = 2
NUM_WORKERS = 16
DEVICE = "cuda"

if __name__ == "__main__":

    SAVE_FILE = "~/instrument_pitch_tracker/windowed/log_dummy.log"

    print("debug - we have the libraries")

    DATA_FOLDER = "./"
    #DATA_FOLDER = "../data/"
    data_path = sorted([os.path.abspath(os.path.join(DATA_FOLDER, f)) for f in os.listdir(DATA_FOLDER)])[0]
    print(data_path)
    dataset = Dataset(os.path.join(data_path, "audio_stems"), os.path.join(data_path, "annotation_stems"))
    print("debug - dataset import success")

    train_set, dev_set, test_set = partition_dataset(dataset, dev_ratio=0.2, test_ratio=0.2)
    del dataset
    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, prefetch_factor=8,
                                   shuffle=True, collate_fn=Collator(BATCH_TRACKS, shuffle=True))
    dev_loader = data.DataLoader(dev_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, prefetch_factor=8,
                                 shuffle=False, collate_fn=Collator(BATCH_TRACKS, shuffle=False))

    model = CREPE(pretrained=False).to(DEVICE)
    device = model.linear.weight.device
    print(device)
    criterion = nn.BCELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)
    print("debug - model import success")

    log_file = {'epoch': {'train': [], 'dev': []}, 'batch': {'train': [], 'dev': []}}
    for epoch in range(MAX_EPOCH):
        model = model.train()
        epoch_train_loss = 0
        epoch_dev_loss = 0
        train_loss = 0

        for e, (s, l, _) in enumerate(train_loader):
            print('debug- tracks loaded')
            model = model.train()
            for i, sequence in enumerate(s):
                sequence = sequence.to(device)
                label = l[i].to(device)
                act = model.forward(sequence)
                loss = criterion(act, label)
                train_loss += loss.item()
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if not e % 2:
                torch.cuda.empty_cache()
                train_loss /= 2
                model = model.eval()
                dev_loss = 0
                eval_data = {"ref": [], "est": []}
                with torch.set_grad_enabled(False):
                    for (s, l, f) in dev_loader:
                        for i, sequence in enumerate(s):
                            sequence = sequence.to(device)
                            label = l[i].to(device)
                            act = model.forward(sequence)
                            loss = criterion(act, label)
                            dev_loss += loss.item()

                            est_hz = to_freq(act, viterbi=False).numpy()
                            ref_hz = f.view(-1).numpy()
                            # confidence = act.max(dim=1)[0][mask].numpy()
                            eval_data["ref"].append(ref_hz)
                            eval_data["est"].append(est_hz)
                torch.cuda.empty_cache()
                dev_loss /= dev_set.__len__()
                ref_hz = np.concatenate(eval_data["ref"])
                est_hz = np.concatenate(eval_data["est"])
                print('epoch: {}  '.format(epoch) + 'trainL: {:.2f}  devL: {:.2f}  '.format(train_loss, dev_loss) +
                      '    '.join('{}: {:.2f}'.format(k, v) for k, v in eval_from_hz(ref_hz, est_hz).items()))

                log_file['batch']['train'].append(train_loss)
                log_file['batch']['dev'].append(dev_loss)
                epoch_train_loss += train_loss
                epoch_dev_loss += dev_loss
                train_loss = 0
        log_file['epoch']['train'].append(epoch_train_loss)
        log_file['epoch']['dev'].append(epoch_dev_loss)
        with open(SAVE_FILE, 'wb') as fp:
            pickle.dump(log_file, fp, protocol=pickle.HIGHEST_PROTOCOL)


