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
from tqdm import tqdm

from models import VQVAE, BinaryFileSource, LBG
from loss_func import calc_lf0_rmse, vqvae_loss
from util import create_loader, train, test, parse
from synthesize import gen_waveform

device = "cuda" if torch.cuda.is_available() else "cpu"

use_attention = False
enable_quantize = True


def train_vqvae(args, trial=None):
    """
    """
    model = VQVAE(
        num_layers=args["num_layers"], z_dim=args["z_dim"], num_class=args["num_class"], 
        input_linguistic_dim = 442+2, enable_quantize=enable_quantize
    ).to(device)

    train_loader, test_loader = create_loader()# 285 ~ 334, 488 ~ 530

    length_osaka = len(train_loader)

    train_loader = np.concatenate([train_loader, train_loader_tokyo])
    
    np.random.shuffle(train_loader)
    if args["model_path"] != "":
        #model.load_state_dict(torch.load("osaka_vqvae_lr1e-5/vqvae_model_20.pth"))
        model.load_state_dict(torch.load(args["model_path"]))

    elif args['num_epoch'] != 0:

        lbg = LBG(num_class=args["num_class"], z_dim=args["z_dim"])
        # zを用意

        sampled_indices = random.sample(
            list(range(len(train_loader))), min(len(train_loader), 2000)
        )
        z = torch.tensor([[0.0] * args["z_dim"]]).to(device)

        print("コードブックを初期化")
        for index in tqdm(sampled_indices):
            data = train_loader[index]
            #print(data[0].shape, data[1].shape, data[2])
            with torch.no_grad():
                z_tmp = model.encode(
                    torch.tensor(data[0]).float().to(device),
                    torch.tensor(data[1]).float().to(device),
                    data[2],
                    tokyo = (data[0][:, -1][0] == 1)
                ).view(-1, args["z_dim"])
                z = torch.cat([z, z_tmp], dim=0).to(device)
        init_codebook = torch.from_numpy(lbg.calc_q_vec(z)).to(device)
        print(init_codebook)
        codebook = nn.Parameter(init_codebook)
        model.init_codebook(codebook)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)  # 1e-3
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 0.93 ** epoch)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    loss_list = []
    train_f0loss_list = []
    test_loss_list = []
    f0_loss_list = []
    start = time.time()
    for epoch in range(1, args["num_epoch"] + 1):
        loss, train_f0_loss = train(
            epoch, model, train_loader, vqvae_loss, optimizer, f0=True
        )
        test_loss, f0_loss = test(epoch, model, test_loader, vqvae_loss)

        print(
            "epoch [{}/{}], loss: {:.4f} test_loss: {:.4f}".format(
                epoch + 1, args["num_epoch"], loss, test_loss
            )
        )
        # scheduler.step()
        # logging
        loss_list.append(loss)
        train_f0loss_list.append(train_f0_loss)
        test_loss_list.append(test_loss)
        f0_loss_list.append(f0_loss)

        if trial is not None:
            trial.report(test_loss, epoch - 1)

        if trial is not None:
            if trial.should_prune():
                return optuna.TrialPruned()

        print(time.time() - start)

        if epoch % 5 == 0:
            torch.save(
                model.state_dict(),
                args["output_dir"] + "/vqvae_model_{}.pth".format(epoch),
            )
        np.savetxt(args["output_dir"] + "/loss_list.csv", np.array(loss_list))
        np.savetxt(args["output_dir"] + "/f0loss_list.csv", np.array(train_f0loss_list))
        np.savetxt(args["output_dir"] + "/test_loss_list.csv", np.array(test_loss_list))
        np.savetxt(args["output_dir"] + "/test_f0loss_list.csv", np.array(f0_loss_list))
        torch.save(
                model.state_dict(),
                args["output_dir"] + "/vqvae_model_tmp.pth".format(epoch),
            )
        

    if args["num_epoch"] == 0:
        f0_loss = 0
        print('サンプル合成フェーズ')
        train_z = []
        valid_z = []
        test_z = []
        train_loader, valid_loader = create_loader(valid=True)
        train_loader, test_loader = create_loader(valid=False)



        model.eval()
    
        
        for i, data in tqdm(enumerate(test_loader)):
            tmp = []
            for j in range(2):
                tmp.append(torch.tensor(data[j]).float().to(device))

            recon_batch, z_mu, z_unquantized_logvar = model(tmp[0], tmp[1], data[2], 0, tokyo=data[0][:, -1][0] == 1)
            #recon_batch = model.decode(torch.zeros(z_mu.size()), tmp[0], data[2], tokyo=data[0][:, -1][0] == 1)
            test_z.append(z_mu.detach().cpu().numpy())
            recon_f0 = recon_batch.detach().cpu().numpy().reshape(-1)
            np.savetxt(join(args['output_dir'], 'generated/BASIC5000_{:04d}.csv'.format(i*13+2)), recon_f0)
            lf0_start_idx = 180
            y_base = data[1]
            y = np.concatenate([y_base[:, :lf0_start_idx], recon_f0.reshape(-1, 1), np.zeros((y_base.shape[0], 2)), y_base[:, lf0_start_idx+3:]], axis=1)
            waveform = gen_waveform(y, True, verbose=False)
            ratio = 28000 / max(waveform.max(), abs(waveform.min()))
            wavfile.write(join(args['output_dir'], 'generated/BASIC5000_{}{}.wav'.format('0'*(4-len(str(i*13+2))), i*13+2)), rate=48000, data=(waveform*ratio).astype(np.int16))

            del tmp
            f0_loss += np.sqrt(((recon_f0 - y_base[:, lf0_start_idx].reshape(-1))**2).mean())
        print(f0_loss*1200/np.log(2)/len(test_loader))

        f0_loss = 0
        for i, data in enumerate(train_loader):
            tmp = []
            for j in range(2):
                tmp.append(torch.tensor(data[j]).float().to(device))
            recon_batch, z_mu, z_unquantized_logvar = model(tmp[0], tmp[1], data[2], 0, tokyo=data[0][:, -1][0] == 1)
            recon_f0 = recon_batch.detach().cpu().numpy().reshape(-1)
            train_z.append(z_mu.detach().cpu().numpy())
            f0_loss += np.sqrt(((recon_f0 - data[1][:, lf0_start_idx].reshape(-1))**2).mean())
            del tmp
        print(f0_loss*1200/np.log(2)/len(train_loader))
        
        f0_loss = 0
        for i, data in enumerate(valid_loader):
            tmp = []
            for j in range(2):
                tmp.append(torch.tensor(data[j]).float().to(device))
            recon_batch, z_mu, z_unquantized_logvar = model(tmp[0], tmp[1], data[2], 0, tokyo=data[0][:, -1][0] == 1)
            recon_f0 = recon_batch.detach().cpu().numpy().reshape(-1)
            valid_z.append(z_mu.detach().cpu().numpy())
            del tmp
            f0_loss += np.sqrt(((recon_f0 - data[1][:, lf0_start_idx].reshape(-1))**2).mean())
        print(f0_loss*1200/np.log(2)/len(valid_loader))
        
        

    
        with open(join(args['output_dir'], 'train_z.pickle'), 'wb') as f:
            pickle.dump(train_z, f)
        with open(join(args['output_dir'], 'valid_z.pickle'), 'wb') as f:
            pickle.dump(valid_z, f)
        
        with open(join(args['output_dir'], 'test_z.pickle'), 'wb') as f:
            pickle.dump(test_z, f)

   
        return 


    return f0_loss


if __name__ == "__main__":
    args = parse()
    os.makedirs(args.output_dir, exist_ok=True)
    train_vqvae(vars(args))
