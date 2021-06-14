import numpy as np
from modules.crepe import CREPE, get_frame
from modules.utils import to_freq, eval_from_hz
from torch.utils import data
import torch.nn as nn
import torch
from modules.dataset import Collator, DictDataset, partition_dataset
import pickle

LEARNING_RATE = 1e-6
MAX_EPOCH = 200
BATCH_SIZE = 1024
BATCH_TRACKS = 16
NUM_WORKERS = 8
DEVICE = "cuda"

if __name__ == "__main__":

    SAVE_FILE = "~/instrument_pitch_tracker/windowed/log_dummy.log"

    print("debug - we have the libraries")

    dataset = DictDataset("/homedtic/ntamer/instrument_pitch_tracker/data/MDB-stem-synth/prep")
    print("debug - dataset import success")

    train_set, dev_set, test_set = partition_dataset(dataset, dev_ratio=0.2, test_ratio=0.2)
    print(train_set.__len__(), dev_set.__len__(), test_set.__len__())
    del dataset
    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, prefetch_factor=2,
                                   shuffle=True, collate_fn=Collator(BATCH_TRACKS, shuffle=True))
    dev_loader = data.DataLoader(dev_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, prefetch_factor=2,
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
            print('debug - tracks loaded')
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
                print('success')

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
                dev_loss /= len(dev_set)
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


