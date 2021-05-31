import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import time
import numpy as np
from tqdm import tnrange, tqdm
import optuna
import os
import random
from os.path import join
from scipy.io import wavfile
import pickle
import pandas as pd


from models import Rnn, BinaryFileSource, LBG
from loss_func import calc_lf0_rmse, vqvae_loss, rmse
from util import create_loader, train_rnn, test_rnn, parse, create_loader_duration
from synthesize import gen_waveform


from nnmnkwii.datasets import FileDataSource, FileSourceDataset
from nnmnkwii.datasets import PaddedFileSourceDataset, MemoryCacheDataset
from nnmnkwii.preprocessing import trim_zeros_frames, remove_zeros_frames
from nnmnkwii.preprocessing import minmax, meanvar, minmax_scale, scale
from nnmnkwii import paramgen
from nnmnkwii.io import hts
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.postfilters import merlin_post_filter

binary_dict, continuous_dict = hts.load_question_set("./data/questions_jp_w_accent.hed")
x_stats = pd.read_csv('./data/x_stats.csv')
acoustic_subphone_features = 'coarse_coding'


device = "cuda" if torch.cuda.is_available() else "cpu"

w_accent = False

def train_vqvae(args, trial=None):
    """
    """
    model = Rnn(
        num_layers=args["num_layers"], input_linguistic_dim = 531+2 if w_accent else 442+2
    ).to(device)

    train_loader, test_loader = create_loader_duration()# 285 ~ 334, 488 ~ 530
    train_loader_tokyo, _ = create_loader_duration(tokyo=True, appendix='_with_acc')
    train_loader_tokyo = train_loader_tokyo[:3000]

    length_osaka = len(train_loader)

    #np.random.seed(0)
    #np.random.shuffle(train_loader)
    #train_num = 1000#int(args["train_ratio"] * len(train_loader))  # 1
    #train_loader = train_loader[:train_num]

    train_loader = np.concatenate([train_loader, train_loader_tokyo])

    train_num = int(args["train_ratio"] * len(train_loader))  # 1
    train_loader = train_loader[:train_num]


    for i in range(len(train_loader)):
        if w_accent:
            train_loader[i][0] = np.concatenate((train_loader[i][0], np.ones((train_loader[i][0].shape[0], 1)), np.zeros((train_loader[i][0].shape[0], 1))), axis=1).astype('float64') if i < 3696 else np.concatenate((train_loader[i][0], np.zeros((train_loader[i][0].shape[0], 1)), np.ones((train_loader[i][0].shape[0], 1))), axis=1).astype('float64')
        else:
            train_loader[i][0] = np.concatenate((train_loader[i][0][:, :285], train_loader[i][0][:, 335:488], train_loader[i][0][:, -4:], np.ones((train_loader[i][0].shape[0], 1)), np.zeros((train_loader[i][0].shape[0], 1))), axis=1).astype('float64')
    
    for i in range(len(test_loader)):
        if w_accent:
            test_loader[i][0] = np.concatenate((test_loader[i][0], np.ones((test_loader[i][0].shape[0], 1)), np.zeros((test_loader[i][0].shape[0], 1))), axis=1).astype('float64')
        else:
            test_loader[i][0] = np.concatenate((test_loader[i][0][:, :285], test_loader[i][0][:, 335:488], test_loader[i][0][:, -4:], np.ones((test_loader[i][0].shape[0], 1)), np.zeros((test_loader[i][0].shape[0], 1))), axis=1).astype('float64')
    
    np.random.shuffle(train_loader)
    if args["model_path"] != "":
        model.load_state_dict(torch.load(args["model_path"]))


    optimizer = optim.Adam(model.parameters(), lr=3e-6)# 1e-3
    loss_list = []
    train_f0loss_list = []
    test_loss_list = []
    f0_loss_list = []
    start = time.time()
    for epoch in range(1, args["num_epoch"] + 1):
        loss, train_f0_loss = train_rnn(
            epoch, model, train_loader, optimizer, duration=True
        )
        test_loss, f0_loss = test_rnn(epoch, model, test_loader, duration=True)
        print(f0_loss)

        print(
            "epoch [{}/{}], loss: {:.4f} test_loss: {:.4f}".format(
                epoch + 1, args["num_epoch"], loss, test_loss
            )
        )
        # scheduler.step()
        # logging
        loss_list.append(loss)
        test_loss_list.append(test_loss)

        if trial is not None:
            trial.report(test_loss, epoch - 1)

        if trial is not None:
            if trial.should_prune():
                return optuna.TrialPruned()

        print(time.time() - start)

        if epoch % 5 == 0:
            torch.save(
                model.state_dict(),
                args["output_dir"] + "/rnn_model_{}.pth".format(epoch),
            )
        np.savetxt(args["output_dir"] + "/loss_list.csv", np.array(loss_list))
        np.savetxt(args["output_dir"] + "/f0loss_list.csv", np.array(train_f0loss_list))
        np.savetxt(args["output_dir"] + "/test_loss_list.csv", np.array(test_loss_list))
        np.savetxt(args["output_dir"] + "/test_f0loss_list.csv", np.array(f0_loss_list))

    if args["num_epoch"] == 0:
        model.load_state_dict(torch.load(args["model_path"]))
        test_z = []
        train_loader, test_loader = create_loader_duration(valid=False)
        for i in range(len(test_loader)):
            if w_accent:
                test_loader[i][0] = np.concatenate((test_loader[i][0], np.ones((test_loader[i][0].shape[0], 1)), np.zeros((test_loader[i][0].shape[0], 1))), axis=1).astype('float64')
            else:
                test_loader[i][0] = np.concatenate((test_loader[i][0][:, :285], test_loader[i][0][:, 335:488], test_loader[i][0][:, -4:], np.ones((test_loader[i][0].shape[0], 1)), np.zeros((test_loader[i][0].shape[0], 1))), axis=1).astype('float64')
    
    
        model.eval()
        for i, data in tqdm(enumerate(test_loader)):
            tmp = []
            for j in range(2):
                tmp.append(torch.tensor(data[j]).float().to(device))

            recon_batch = model(tmp[0])
            recon_duration = recon_batch.detach().cpu().numpy().reshape(-1).astype('int64')
            recon_duration[recon_duration <= 0] = 1

            hts_label = hts.load('data/label_phone_align/OSAKA_PHRASES3696_{}{}.lab'.format('0'*(4-len(str(i*13+2))), i*13+2))

            recon_duration = np.concatenate([[0], recon_duration, [0]])
            hts_label.set_durations(recon_duration)

            lin_features = fe.linguistic_features(hts_label,
                                                            binary_dict, continuous_dict,
                                                            add_frame_features=True,
                                                            subphone_features=acoustic_subphone_features)

            indices = hts_label.silence_frame_indices()
            lin_features = np.delete(lin_features, indices, axis=0)
            lin_features = minmax_scale(lin_features, x_stats['min'].values, x_stats['max'].values, feature_range=(0.01, 0.99))

            fname = join(args['output_dir'], 'predicted_X_acoustic', 'OSAKA_PHRASES3696_{}{}.csv'.format('0'*(4-len(str(i*13+2))), i*13+2))
            np.savetxt(fname, lin_features)

            """

            tmp_data = []
            for linguistic_features, duration in zip(data[0], recon_duration):
                for i in range(duration):
                    tmp_data.append(linguistic_features)
            
            np.save(join(args['output_dir'],  'X_acoustic_predicted/OSAKA3696_{}{}.bin'.format('0'*(4-len(str(i*13+2))), i*13+2)), tmp_data)


            np.savetxt(join(args['output_dir'], 'generated/OSAKA3696_{}{}.csv'.format('0'*(4-len(str(i*13+2))), i*13+2)), recon_duration)
            """

            del tmp


    return 


if __name__ == "__main__":
    args = parse()
    os.makedirs(args.output_dir, exist_ok=True)
    train_vqvae(vars(args))