import torch
import numpy as np
import os
from mir_eval import melody


def eval_pitch_std(ref_voicing, ref_cent, est_voicing, est_cent, cent_tolerance=50):
    diff = est_cent - ref_cent
    voicing_zone = np.logical_and(ref_voicing, est_voicing)
    tolerance_zone = abs(diff) <= cent_tolerance
    zone = np.logical_and(voicing_zone, tolerance_zone)
    return np.std(diff[zone])


def eval_from_hz(ref_hz, est_hz):
    ref_cent = melody.hz2cents(ref_hz)
    est_cent = melody.hz2cents(est_hz)
    ref_voicing = np.array(ref_cent > 0, dtype=float)
    est_voicing = np.array(est_cent > 0, dtype=float)
    recall, false_alarm = melody.voicing_measures(ref_voicing, est_voicing)
    return {"recall": recall, "false_alarm": false_alarm,
            "rca50": melody.raw_chroma_accuracy(ref_voicing, ref_cent, est_voicing, est_cent, cent_tolerance=50),
            "rpa50": melody.raw_pitch_accuracy(ref_voicing, ref_cent, est_voicing, est_cent, cent_tolerance=50),
            "rpa25": melody.raw_pitch_accuracy(ref_voicing, ref_cent, est_voicing, est_cent, cent_tolerance=25),
            "rpa10": melody.raw_pitch_accuracy(ref_voicing, ref_cent, est_voicing, est_cent, cent_tolerance=10),
            "rpa5": melody.raw_pitch_accuracy(ref_voicing, ref_cent, est_voicing, est_cent, cent_tolerance=5),
            "std50": eval_pitch_std(ref_voicing, ref_cent, est_voicing, est_cent, cent_tolerance=50)}


def download_weights(model_capacity):
    try:
        from urllib.request import urlretrieve
    except ImportError:
        from urllib import urlretrieve

    weight_file = 'crepe-{}.pth'.format(model_capacity)
    base_url = 'https://github.com/sweetcocoa/crepe-pytorch/raw/models/'

    # in all other cases, decompress the weights file if necessary
    package_dir = os.path.dirname(os.path.realpath(__file__))
    weight_path = os.path.join(package_dir, weight_file)
    if not os.path.isfile(weight_path):
        print('Downloading weight file {} from {} ...'.format(weight_path, base_url + weight_file))
        urlretrieve(base_url + weight_file, weight_path)


def to_local_average_cents(salience, center=None):
    """
    find the weighted average cents near the argmax bin
    """

    if not hasattr(to_local_average_cents, 'cents_mapping'):
        # the bin number-to-cents mapping
        to_local_average_cents.mapping = (
                torch.tensor(np.linspace(0, 7180, 360)) + 1997.3794084376191)

    if isinstance(salience, np.ndarray):
        salience = torch.from_numpy(salience)

    if salience.ndim == 1:
        if center is None:
            center = int(torch.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = torch.sum(
            salience * to_local_average_cents.mapping[start:end].to(salience.device))
        weight_sum = torch.sum(salience)
        return product_sum / weight_sum
    if salience.ndim == 2:
        return torch.tensor([to_local_average_cents(salience[i, :]) for i in
                             range(salience.shape[0])])

    raise Exception("label should be either 1d or 2d Tensor")


def to_viterbi_cents(salience):
    """
    Find the Viterbi path using a transition prior that induces pitch
    continuity.

    * Note : This is NOT implemented with pytorch.
    """
    from hmmlearn import hmm

    # uniform prior on the starting pitch
    starting = np.ones(360) / 360

    # transition probabilities inducing continuous pitch
    xx, yy = np.meshgrid(range(360), range(360))
    transition = np.maximum(12 - abs(xx - yy), 0)
    transition = transition / np.sum(transition, axis=1)[:, None]

    # emission probability = fixed probability for self, evenly distribute the
    # others
    self_emission = 0.1
    emission = (np.eye(360) * self_emission + np.ones(shape=(360, 360)) *
                ((1 - self_emission) / 360))

    # fix the model parameters because we are not optimizing the model
    model = hmm.MultinomialHMM(360, starting, transition)
    model.startprob_, model.transmat_, model.emissionprob_ = \
        starting, transition, emission

    # find the Viterbi path
    observations = np.argmax(salience, axis=1)
    path = model.predict(observations.reshape(-1, 1), [len(observations)])

    return np.array([to_local_average_cents(salience[i, :], path[i]) for i in
                     range(len(observations))])


def to_freq(activation, viterbi=False):
    if viterbi:
        cents = to_viterbi_cents(activation.detach().numpy())
        cents = torch.tensor(cents)
    else:
        cents = to_local_average_cents(activation)

    frequency = 10 * 2 ** (cents / 1200)
    frequency[torch.isnan(frequency)] = 0
    return frequency


def print_model_info(model_id, args, writer):
    model_h_params = ""
    for x in args:
        params = ""
        try:
            for y in args[x]:
                params += y + ": " + str(args[x][y]) + "\n"
        except TypeError:
            params += "%s" % args[x] + "\n"
        model_h_params += "\n      _" + x + "_\n" + params
    print("model ID:", model_id, "\n", model_h_params)
    writer.add_text(model_id, model_h_params, global_step=0)
    writer.flush()
    return writer


def evaluate(dev_loader, model, criterion):
    device = model.linear.weight.device
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

                est_hz = to_freq(act, viterbi=False).view(-1).numpy()
                ref_hz = f[i].view(-1).numpy()
                # confidence = act.max(dim=1)[0][mask].numpy()
                eval_data["ref"].append(ref_hz)
                eval_data["est"].append(est_hz)
    torch.cuda.empty_cache()
    ref_hz = np.concatenate(eval_data["ref"])
    est_hz = np.concatenate(eval_data["est"])

    return dev_loss, eval_from_hz(ref_hz, est_hz)
