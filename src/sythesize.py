from os.path import expanduser, join


import sys

import time

from nnmnkwii.datasets import FileDataSource, FileSourceDataset
from nnmnkwii.datasets import PaddedFileSourceDataset, MemoryCacheDataset  # これはなに？
from nnmnkwii.preprocessing import trim_zeros_frames, remove_zeros_frames
from nnmnkwii.preprocessing import minmax, meanvar, minmax_scale, scale
from nnmnkwii import paramgen
from nnmnkwii.io import hts
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.postfilters import merlin_post_filter

from os.path import join, expanduser, basename, splitext, basename, exists
import os
from glob import glob
import numpy as np
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
import pyworld
import pysptk
import librosa
import librosa.display
import IPython
from IPython.display import Audio

import matplotlib.pyplot as plt

from torch.utils import data as data_utils


import torch
from torch import nn
from torch.autograd import Variable
from tqdm import tnrange, tqdm
from torch import optim
import torch.nn.functional as F

mgc_dim = 180  # メルケプストラム次数　？？
lf0_dim = 3  # 対数fo　？？ なんで次元が３？
vuv_dim = 1  # 無声or 有声フラグ　？？
bap_dim = 15  # 発話ごと非周期成分　？？

duration_linguistic_dim = 438  # question_jp.hed で、ラベルに対する言語特徴量をルールベースで記述してる
acoustic_linguisic_dim = 442  # 上のやつ+frame_features とは？？
duration_dim = 1
acoustic_dim = mgc_dim + lf0_dim + vuv_dim + bap_dim  # aoustice modelで求めたいもの

fs = 48000
frame_period = 5
fftlen = pyworld.get_cheaptrick_fft_size(fs)
alpha = pysptk.util.mcepalpha(fs)
hop_length = int(0.001 * frame_period * fs)

mgc_start_idx = 0
lf0_start_idx = 180
vuv_start_idx = 183
bap_start_idx = 184

windows = [
    (0, 0, np.array([1.0])),
    (1, 1, np.array([-0.5, 0.0, 0.5])),
    (1, 1, np.array([1.0, -2.0, 1.0])),
]

use_phone_alignment = True
acoustic_subphone_features = "coarse_coding" if use_phone_alignment else "full"  # とは？

model_path = '0929_tokyo3000included_alllabeled_lr1e-6/vqvae_model_40.pth'

from models import VQVAE

device =  'cuda'
#model = VQVAE(num_layers=2, z_dim=1, num_class=4, input_linguistic_dim = 289+2).to(device)#289+2
#model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))

#model = VQVAE(num_layers=2, z_dim=1, num_class=2, input_linguistic_dim = 289+2).to(device)#289+2
#model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))

#base_model = Rnn(output_dim=196, input_linguistic_dim=442+2).to(device)
#base_model.load_state_dict(torch.load('baseline_nonf0_onlytokyo/rnn_synthesize_model_10.pth', map_location=torch.device('cuda')))



from copy import deepcopy
from util import create_loader
train_loader, test_loader = create_loader(valid=False)
test_loader_for_base = deepcopy(test_loader)

for i in range(len(test_loader)):
    test_loader[i][0] = np.concatenate((test_loader[i][0][:, :285], test_loader[i][0][:, -4:], np.ones((test_loader[i][0].shape[0], 1)), np.zeros((test_loader[i][0].shape[0], 1))), axis=1)
    test_loader_for_base[i][0] = np.concatenate((test_loader_for_base[i][0], np.ones((test_loader_for_base[i][0].shape[0], 1)), np.zeros((test_loader_for_base[i][0].shape[0], 1))), axis=1)
    test_loader[i][2][-1] = test_loader[i][0].shape[0]-1

def recon_check_z(index, valid=False,):
    model.eval()
    with torch.no_grad():
        tmp = []
        data = test_loader[index]
        if valid:
            data = valid_loader[index]
        for j in range(2):
            tmp.append(torch.from_numpy(data[j]).float().to(device))
  
        y, mu, logvar = model(tmp[0], tmp[1], data[2], 0)


    return mu.view(-1).detach().cpu().numpy()

"""
z = []
device='cuda'
import pickle
model.eval()
for i in tqdm(range(100)):
    z.append(recon_check_z(i))
nc = model.quantized_vectors.weight.size()[0]
f = open('vqvae_z{}.txt'.format(model_path[:model_path.find('/')]), 'wb')
pickle.dump(z, f)
"""

def synthesize(index,verbose=True, z0=None, valid=False, lh=False):
    model.eval()
    y = recon(index, z0=z0, valid=valid)
    y_base = test_loader[index][1].copy()#basemodel(torch.from_numpy(test_loader[index][0])).detach().cpu().numpy().reshape(-1, 199) if not valid else basemodel(torch.from_numpy(valid_loader[index][0])).detach().cpu().numpy().reshape(-1, 199)
    y_base[:, lf0_start_idx] = y.detach().cpu().numpy().reshape(-1)
    y_base[:, lf0_start_idx+1:lf0_start_idx+3] = 0
    if verbose:
        IPython.display.display(Audio(gen_waveform(y_base, True), rate=fs))
    else:
        return y.cpu().numpy().reshape(-1)

def recon(index, z0=None, valid=False, verbose=True):
    model.eval()
    with torch.no_grad():
        tmp = []
        data = test_loader[index]
        if valid:
            data = valid_loader[index]
        for j in range(2):
            tmp.append(torch.from_numpy(data[j]).float().to(device))
            
        if z0 is not None:
            y = model.decode(torch.tensor(z0), tmp[0], data[2], tokyo=False)
            return y

        y, mu, logvar = model(tmp[0], tmp[1], data[2], 0)

        
    return y# mu, logvar


import pandas as pd

y_stats = pd.read_csv('data/y_stats.csv')

def rmse(A, B) :
    return np.sqrt((np.square(A - B)).mean())

def calc_lf0_rmse(natural, generated, lf0_idx=lf0_start_idx, vuv_idx=vuv_start_idx):
    idx = (natural[:, vuv_idx]).astype(bool)
    return rmse(natural[idx, lf0_idx], generated[idx]) * 1200 / np.log(2)  # unit: [cent]

def gen_parameters(y_predicted, verbose=True):
    # Number of time frames
    T = y_predicted.shape[0]
    
    # Split acoustic features
    mgc = y_predicted[:,:lf0_start_idx]
    lf0 = y_predicted[:,lf0_start_idx:vuv_start_idx]
    if verbose:
        plt.plot(lf0[:, 0])
        plt.show()
    #lf0 = Y['acoustic']['train'][90][:, lf0_start_idx:vuv_start_idx]
    #lf0 = np.zeros(lf0.shape)
    vuv = y_predicted[:,vuv_start_idx]

    plt.show()
    bap = y_predicted[:,bap_start_idx:]
    
    # Perform MLPG
    ty = "acoustic"
    mgc_variances = np.tile(y_stats['var'][:lf0_start_idx], (T, 1))#np.tile(np.ones(Y_var[ty][:lf0_start_idx].shape), (T, 1))#
    mgc = paramgen.mlpg(mgc, mgc_variances, windows)
    lf0_variances = np.tile(y_stats['var'][lf0_start_idx:vuv_start_idx], (T,1))#np.tile(np.ones(Y_var[ty][lf0_start_idx:vuv_start_idx].shape), (T,1))#
    lf0 = paramgen.mlpg(lf0, lf0_variances, windows)
    bap_variances = np.tile(y_stats['var'][bap_start_idx:], (T, 1))#np.tile(np.ones(Y_var[ty][bap_start_idx:].shape), (T, 1))#
    bap = paramgen.mlpg(bap, bap_variances, windows)
    
    return mgc, lf0, vuv, bap
def gen_waveform(y_predicted, do_postfilter=True, verbose=False):  
    y_predicted = trim_zeros_frames(y_predicted)
        
    # Generate parameters and split streams
    mgc, lf0, vuv, bap = gen_parameters(y_predicted, verbose=verbose)
    
    if do_postfilter:
        mgc = merlin_post_filter(mgc, alpha)
        
    spectrogram = pysptk.mc2sp(mgc, fftlen=fftlen, alpha=alpha)
    aperiodicity = pyworld.decode_aperiodicity(bap.astype(np.float64), fs, fftlen)
    f0 = lf0.copy()
    f0[vuv < 0.5] = 0
    f0[np.nonzero(f0)] = np.exp(f0[np.nonzero(f0)])
    
    generated_waveform = pyworld.synthesize(f0.flatten().astype(np.float64),
                                            spectrogram.astype(np.float64),
                                            aperiodicity.astype(np.float64),
                                            fs, frame_period)
    return generated_waveform

"""
print('コードブック')
print(model.quantized_vectors.weight)



savedir = './data/generated/' + model_path[:model_path.find('/')] + '_synthesize_nonf0'
f0_e = 0
for i, data in tqdm(enumerate(test_loader_for_base)):
    with torch.no_grad():
        pred_y = recon(i,  verbose=False).cpu().numpy().reshape(-1)
        
        y_base = base_model(torch.from_numpy(data[0]).float().to(device)).detach().cpu().numpy().reshape(-1, 196)

        #y_base[:, lf0_start_idx] = pred_y
        #y_base[:, lf0_start_idx+1:lf0_start_idx+3] = 0
        #y = y_base

        y = np.concatenate([y_base[:, :lf0_start_idx], pred_y.reshape(-1, 1), np.zeros((y_base.shape[0], 2)), y_base[:, lf0_start_idx:]], axis=1)
        waveform = gen_waveform(y, True, verbose=False)
        wavfile.write(join(savedir, 'OSAKA1300_{}{}.wav'.format('0'*(4-len(str(i*13+2))), i*13+2)), rate=fs, data=waveform.astype(np.int16))
        f0_e += calc_lf0_rmse(data[1], pred_y)

print('f0 rmse')
print(f0_e / 100)
"""