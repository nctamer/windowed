import numpy as np
from modules.crepe import CREPE, get_frame
from modules.utils import print_model_info, evaluate
from torch.utils import data
import torch.nn as nn
import torch
from modules.dataset import Collator, DictDataset
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import json
from torch_audiomentations import Compose, Gain, PolarityInversion, AddBackgroundNoise, ApplyImpulseResponse, \
    AddColoredNoise

args = {
    "learning_rate": 4e-4,
    "max_epoch": 200,
    "batch_size": 128,
    "batch_tracks": 512,
    "num_workers": 5,
    "device": "cuda",
    "augment": False,
    "data": "prep2048"
}
model_id = "bare2048"

parent_dir = "/homedtic/ntamer/instrument_pitch_tracker/"

# Initialize augmentation callable
apply_augmentation = Compose(
    transforms=[
        PolarityInversion(p=0.5),
        Gain(
            min_gain_in_db=-15.0,
            max_gain_in_db=5.0,
            p=0.5, mode="per_batch"
        ),
        ApplyImpulseResponse(ir_paths="test_fixtures/ir",
                             p=0.5, mode="per_batch"),
        AddColoredNoise(min_snr_in_db=10.0,
                        max_snr_in_db=30.0,
                        p=0.3, mode="per_batch"),
        AddBackgroundNoise(
            background_paths="test_fixtures/bg_short",
            min_snr_in_db=10.0,
            max_snr_in_db=30.0,
            p=0.2,
            mode="per_batch"
        ),
    ]
)

if __name__ == "__main__":
    models_dir = os.path.join(parent_dir, "models")
    """ dataset & model """
    save_dir = os.path.join(models_dir, model_id + "_" + str(datetime.now().strftime("%b%d_%H")))
    os.makedirs(save_dir, exist_ok=True)
    out_file = os.path.join(save_dir, "out.txt")
    writer = SummaryWriter(log_dir=save_dir, filename_suffix=".board")
    with open(os.path.join(save_dir, 'args.txt'), 'w') as json_file:
        json.dump(args, json_file)
    writer = print_model_info(model_id, args, writer)
    print("args:\n",   '    '.join('{}: {}'.format(k, v) for k, v in args.items()), file=open(out_file, "w"))
    train_set = DictDataset(os.path.join(parent_dir, "data/MDB-stem-synth", args["data"]))
    dev_set = DictDataset(os.path.join(parent_dir, "data/Bach10-mf0-synth", args["data"]), instrument_name="violin")
    test_set = DictDataset(os.path.join(parent_dir, "data/Bach10-mf0-synth", args["data"]))
    # train_set, dev_set, test_set = partition_dataset(dataset, dev_ratio=0.05, test_ratio=0.05)
    print("splits:", train_set.__len__(), dev_set.__len__(), test_set.__len__(), file=open(out_file, "a"))

    train_loader = data.DataLoader(train_set, batch_size=args["batch_tracks"], num_workers=args["num_workers"],
                                   shuffle=True, collate_fn=Collator(args["batch_size"], shuffle=True))
    dev_loader = data.DataLoader(dev_set, batch_size=args["batch_tracks"]//4, num_workers=args["num_workers"],
                                 shuffle=False, collate_fn=Collator(args["batch_size"]*8, shuffle=False))
    criterion = nn.BCELoss(reduction="sum")

    model = CREPE(pretrained=True).to(args["device"]).eval()
    device = model.linear.weight.device
    dev_loss, performance_dict = evaluate(dev_loader, model, criterion)
    print("device:", device, file=open(out_file, "a"))
    print('PRETRAINED:  ' + '  devL: {:.2f}  '.format(dev_loss) +
          '    '.join('{}: {:.3f}'.format(k, v) for k, v in performance_dict.items()),
          file=open(out_file, "a"))

    writer.add_scalars('ORIGINAL PRETRAINED Accuracy', {'RCA50': float(performance_dict["rca50"]),
                                                        'RPA50': float(performance_dict["rpa50"]),
                                                        'RPA25': float(performance_dict["rpa25"]),
                                                        'RPA10': float(performance_dict["rpa10"]),
                                                        'RPA5': float(performance_dict["rpa5"])}, global_step=0)

    # NOW TRAIN THE MODEL FROM SCRATCH
    model = CREPE(pretrained=False).to(args["device"])
    print("\nTraining started\n", file=open(out_file, "a"))

    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"], betas=(0.9, 0.999), eps=1e-8)

    global_step = 0
    best_step, best_dev_loss = 0, np.inf  # to do early stopping if the loss is not getting better
    for epoch in range(args["max_epoch"]):
        for e, (s, l, _) in enumerate(train_loader):

            # train
            train_loss = 0
            model = model.train()
            for i, sequence in enumerate(s):
                sequence = sequence.to(device)
                if args["augment"]:
                    sequence = apply_augmentation(sequence.unsqueeze(1), sample_rate=16000).squeeze(1)
                label = l[i].to(device)
                act = model.forward(sequence)
                loss = criterion(act, label)
                train_loss += loss.item()
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i == int((500*32)/args["batch_size"]):
                    break
            torch.cuda.empty_cache()

            # evaluate
            dev_loss, performance_dict = evaluate(dev_loader, model.eval(), criterion)
            print('step: {}  '.format(global_step) + 'trainL: {:.2f}  devL: {:.2f}  '.format(train_loss, dev_loss) +
                  '    '.join('{}: {:.3f}'.format(k, v) for k, v in performance_dict.items()),
                  file=open(out_file, "a"))

            writer.add_scalars('Accuracy', {'RCA50': float(performance_dict["rca50"]),
                                            'RPA50': float(performance_dict["rpa50"]),
                                            'RPA25': float(performance_dict["rpa25"]),
                                            'RPA10': float(performance_dict["rpa10"]),
                                            'RPA5': float(performance_dict["rpa5"])}, global_step=global_step)
            writer.add_scalar('Cents/Std50', float(performance_dict["std50"]), global_step=global_step)
            writer.add_scalars('Loss',  {'Train': train_loss,
                                         'Dev': dev_loss}, global_step=global_step)
            writer.flush()

            # save the model if there is improvement
            if dev_loss < best_dev_loss:
                torch.save(model.state_dict(), os.path.join(save_dir, model_id))
                best_dev_loss = dev_loss
                best_step = global_step
            elif global_step - best_step > 20:
                print("\nFinished!!!\nBest Step: {}".format(global_step), file=open(out_file, "a"))
                raise SystemExit(0)  # stop if dev loss is not reduced for over many epochs
            global_step += 1
        print("\n Epoch ", epoch, file=open(out_file, "a"))

