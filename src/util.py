import numpy as np
import argparse
from glob import glob
from nnmnkwii.datasets import FileDataSource, FileSourceDataset
from os.path import join
from nnmnkwii.preprocessing import minmax, meanvar, minmax_scale, scale
import torch
import torch.nn.functional as F
from scipy import signal
import copy
import pandas as pd
import pickle

from models import BinaryFileSource
from loss_func import calc_lf0_rmse, rmse

mgc_dim = 180
lf0_dim = 3
vuv_dim = 1
bap_dim = 15
acoustic_linguisic_dim = 442
duration_linguistic_dim = acoustic_linguisic_dim - 4
acoustic_dim = mgc_dim + lf0_dim + vuv_dim + bap_dim
duration_dim = 1
mgc_start_idx = 0
lf0_start_idx = 180
vuv_start_idx = 183
bap_start_idx = 184

device = "cuda:0" if torch.cuda.is_available() else "cpu"#"cpu" #


def parse():
    parser = argparse.ArgumentParser(description="LSTM VAE",)
    parser.add_argument(
        "-ne", "--num_epoch", type=int, default=80,
    )
    parser.add_argument(
        "-nl", "--num_layers", type=int, default=2,
    )
    parser.add_argument(
        "-zd", "--z_dim", type=int, default=1,
    )
    parser.add_argument("-od", "--output_dir", type=str, required=True,),
    parser.add_argument("-dr", "--dropout_ratio", type=float, default=0.3,),
    parser.add_argument("-mp", "--model_path", type=str, default="",),
    parser.add_argument("-tr", "--train_ratio", type=float, default=1.0),
    parser.add_argument("-nc", "--num_class", type=int, default=2),
    parser.add_argument("-q", "--quantized", type=bool, default=False),
    parser.add_argument("-nt", "--num_trials", type=int, default=30)
    parser.add_argument("-ua", "--use_alv", type=bool, default=False)
    parser.add_argument('-s', '--synthesize', type=bool, default=False)

    return parser.parse_args()


def train(epoch, model, train_loader, loss_function, optimizer, f0=False):
    model.train()
    train_loss = 0
    f0_loss = 0
    for batch_idx, data in enumerate(train_loader):
        tmp = []
        for j in range(2):
            tmp.append(torch.from_numpy(data[j]).float().to(device))

        optimizer.zero_grad()
        recon_batch, z_mu, z_unquantized_logvar = model(tmp[0], tmp[1], data[2], epoch, tokyo=data[0][:, -1][0] == 1)
        loss = loss_function(
            recon_batch, tmp[1][:, lf0_start_idx], z_mu, z_unquantized_logvar
        )
        if not torch.isnan(loss):
            loss.backward()
        train_loss += loss.item()
        if f0:
            with torch.no_grad():
                f0_loss += rmse(
                    recon_batch.detach().cpu().numpy().reshape(-1,),
                    tmp[1][:, lf0_start_idx].detach().cpu().numpy().reshape(-1),
                )
        optimizer.step()
        del tmp

    print(
        "====> Epoch: {} Average loss: {:.4f}".format(
            epoch, train_loss / len(train_loader)
        )
    )

    if f0:
        f0_loss /= len(train_loader)
        f0_loss = f0_loss * 1200 / np.log(2)
        return train_loss / len(train_loader), f0_loss
    return train_loss / len(train_loader)


def test(epoch, model, test_loader, loss_function):
    model.eval()
    test_loss = 0
    f0_loss = 0
    with torch.no_grad():
        for i, data, in enumerate(test_loader):
            tmp = []

            for j in range(2):
                tmp.append(torch.tensor(data[j]).float().to(device))

            recon_batch, z_mu, z_unquantized_logvar = model(tmp[0], tmp[1], data[2], 0, tokyo=data[0][:, -1][0] == 1)
            test_loss += loss_function(
                recon_batch, tmp[1][:, lf0_start_idx], z_mu, z_unquantized_logvar
            ).item()
            f0_loss += rmse(
                recon_batch.cpu().numpy().reshape(-1,),
                tmp[1][:, lf0_start_idx].cpu().numpy().reshape(-1),
            )
            del tmp

    test_loss /= len(test_loader)
    f0_loss /= len(test_loader)
    f0_loss = f0_loss * 1200 / np.log(2)
    print("====> Test set loss: {:.4f}".format(test_loss))

    return test_loss, f0_loss

def smooth_f0(y):
    ans = y.copy()
    ans[:, lf0_start_idx] = trajectory_smoothing(
        y[:, lf0_start_idx].reshape(-1, 1)
    ).reshape(-1)
    return ans

def replace_last_mora_i(length, mora_i):
    mora_i[-1]=length-1
    return mora_i

def extract_nearest_code(codebook, z):
    codebook_repeated = codebook.repeat(z.size()[0], 1)
    diff = codebook_repeated.reshape(-1, codebook.size()[0]) - z.reshape(-1, 1)
    indices = torch.argmin(diff * diff, dim=1)
    return codebook[indices]

def z2train_label(codebook, z):
    n_class = codebook.size()[0]
    codebook_repeated = codebook.repeat(z.size()[0], 1)
    diff = codebook_repeated.reshape(-1, n_class) - z.reshape(-1, 1)
    indices = torch.argmin(diff * diff, dim=1)
    codebook_onehot = torch.eye(n_class, n_class)
    codebook_onehot[indices]
    return codebook_onehot[indices].to(device)

def z2index(codebook, z):
    n_class = codebook.size()[0]
    codebook_repeated = codebook.repeat(z.size()[0], 1)
    diff = codebook_repeated.reshape(-1, n_class) - z.reshape(-1, 1)
    indices = torch.argmin(diff * diff, dim=1)
    return indices


def output_z2input_z(codebook, output_z):
    n_class = codebook.size()[0]
    indices = torch.argmax(output_z.reshape(-1, n_class), dim=1)
    return codebook[indices]


def create_loader(valid=True, batch_size=1, tokyo=False, appendix='', accent_phrase = False):
    DATA_ROOT = "data/"

    acoustic_linguisic_dim = 535 if not tokyo else 442
    duration_linguistic_dim = acoustic_linguisic_dim - 4
    X = {"acoustic": {}}
    Y = {"acoustic": {}}
    utt_lengths = {"acoustic": {}}
    for ty in ["acoustic"]:
        for phase in ["train", "test"]:
            train = phase == "train"
            x_dim = (
                acoustic_linguisic_dim if appendix == '' else 535
            )
            y_dim = duration_dim if ty == "duration" else acoustic_dim
            X[ty][phase] = FileSourceDataset(
                BinaryFileSource(
                    join(DATA_ROOT, "X_{}{}".format(ty, appendix)), dim=x_dim, train=train, valid=valid, tokyo=tokyo
                )
            )
            Y[ty][phase] = FileSourceDataset(
                BinaryFileSource(
                    join(DATA_ROOT, "Y_{}{}".format(ty, appendix)), dim=y_dim, train=train, valid=valid, tokyo=tokyo
                )
            )
            utt_lengths[ty][phase] = np.array(
                [len(x) for x in X[ty][phase]], dtype=np.int
            )

    X_min = {}
    X_max = {}
    Y_mean = {}
    Y_var = {}
    Y_scale = {}

    for typ in ["acoustic"]:
        X_min[typ], X_max[typ] = minmax(X[typ]["train"], utt_lengths[typ]["train"])
        Y_mean[typ], Y_var[typ] = meanvar(Y[typ]["train"], utt_lengths[typ]["train"])
        Y_scale[typ] = np.sqrt(Y_var[typ])

    pd.DataFrame({"max": X_max["acoustic"], "min": X_min["acoustic"],}).to_csv(
        "data/x_stats.csv", index=None
    )

    pd.DataFrame(
        {
            "mean": Y_mean["acoustic"],
            "var": Y_var["acoustic"],
            "scale": Y_scale["acoustic"],
        }
    ).to_csv('data/y_stats.csv')

    mora_index_lists = sorted(glob(join("data/mora_index", "squeezed_*.csv"))) if not tokyo else sorted(glob(join(DATA_ROOT, "mora_index/squeezed_*.csv")))

    mora_index_lists_for_model = [
        np.loadtxt(path).reshape(-1) for path in mora_index_lists
    ]

    train_mora_index_lists = []
    test_mora_index_lists = []

    for i, mora_i in enumerate(mora_index_lists_for_model):
        if (i - 1) % 20 == 0:  # test
            if not valid:
	        test_mora_index_lists.append(mora_i)
	elif i % 20 == 0:  # valid
            if valid:
                test_mora_index_lists.append(mora_i)
        else:
            train_mora_index_lists.append(mora_i)

    X_acoustic_train = [
        minmax_scale(
            X["acoustic"]["train"][i],
            X_min["acoustic"],
            X_max["acoustic"],
            feature_range=(0.01, 0.99),
        )
        for i in range(len(X["acoustic"]["train"]))
    ]

    Y_acoustic_train = [smooth_f0(y) for y in Y["acoustic"]["train"]]
    train_mora_index_lists = [
        train_mora_index_lists[i] for i in range(len(train_mora_index_lists))
    ]

    X_acoustic_test = [
        minmax_scale(
            X["acoustic"]["test"][i],
            X_min["acoustic"],
            X_max["acoustic"],
            feature_range=(0.01, 0.99),
        )
        for i in range(len(X["acoustic"]["test"]))
    ]
    Y_acoustic_test = [smooth_f0(y) for y in Y["acoustic"]["test"]]

    # Y_acoustic_test = [scale(Y["acoustic"]["test"][i], Y_mean["acoustic"], Y_scale["acoustic"])for i in range(len(Y["acoustic"]["test"]))]
    test_mora_index_lists = [
        test_mora_index_lists[i] for i in range(len(test_mora_index_lists))
    ]

    train_loader = [
        [X_acoustic_train[i], Y_acoustic_train[i], replace_last_mora_i(X_acoustic_train[i].shape[0], train_mora_index_lists[i])]
        for i in range(len(X_acoustic_train))
    ]
    test_loader = [
        [X_acoustic_test[i], Y_acoustic_test[i], replace_last_mora_i(X_acoustic_test[i].shape[0], test_mora_index_lists[i])]
        for i in range(len(X_acoustic_test))
    ]

    return train_loader, test_loader



def trajectory_smoothing(x, thresh=0.1):
    """apply trajectory smoothing to an array.
    Parameters
    ----------
    x: array, [T, D]
        array to be smoothed
    thresh: scalar, 0 <= thresh <= 1,
        threshold of the trajectory smoothing 
    ------
    y : array, [T, D],
        trajectory-smoothed array
    """

    y = copy.copy(x)

    b, a = signal.butter(2, thresh)
    for d in range(y.shape[1]):
        y[:, d] = signal.filtfilt(b, a, y[:, d])
        y[:, d] = signal.filtfilt(b, a, y[::-1, d])[::-1]

    return y


def num2str(num):
    return "0" * (4 - len(str(num + 1))) + str(num + 1)

