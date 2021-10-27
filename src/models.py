import torch
from torch import nn
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
import numpy as np
from nnmnkwii.datasets import FileDataSource, FileSourceDataset
from os.path import join, expanduser, basename, splitext, basename, exists
import os
from glob import glob
from sklearn.cluster import KMeans
import random

from os import getcwd
import sys
sys.path.append(getcwd())
import hparams


from attention.modules import MultiHeadAttention, PositionwiseFeedForward

mgc_dim = 180  # メルケプストラム次数　？？
lf0_dim = 3  # 対数fo　？？ なんで次元が３？
vuv_dim = 1  # 無声or 有声フラグ　？？
bap_dim = 15  # 発話ごと非周期成分　？？

duration_linguistic_dim = 438  # question_jp.hed で、ラベルに対する言語特徴量をルールベースで記述してる
acoustic_linguisic_dim = 442  # 上のやつ+frame_features とは？？
duration_dim = 1
acoustic_dim = mgc_dim + lf0_dim + vuv_dim + bap_dim  # aoustice modelで求めたいもの

mgc_start_idx = 0
lf0_start_idx = 180
vuv_start_idx = 183
bap_start_idx = 184

windows = [
    (0, 0, np.array([1.0])),
    (1, 1, np.array([-0.5, 0.0, 0.5])),
    (1, 1, np.array([1.0, -2.0, 1.0])),
]


device = "cuda:0" if torch.cuda.is_available() else "cpu"#"cpu" #

hidden_num = 511


class VQVAE(nn.Module):#attentionを導入する
    def __init__(
        self, bidirectional=True, num_layers=2, num_class=2, z_dim=1, dropout=0.15, input_linguistic_dim=acoustic_linguisic_dim, 
        use_attention = False, n_head=1, enable_quantize = True
    ):
        super(VQVAE, self).__init__()
        self.num_layers = num_layers
        self.num_direction = 2 if bidirectional else 1
        self.num_class = num_class
        self.quantized_vectors = nn.Embedding(
            num_class, z_dim
        )
        self.enable_quantize = enable_quantize
        self.quantized_vectors.weight = nn.init.normal_(
            self.quantized_vectors.weight, 0.1, 0.001
        )

        self.z_dim = z_dim

        self.fc11 = nn.Linear(
            input_linguistic_dim + acoustic_dim, input_linguistic_dim + acoustic_dim - 2
        )

        self.lstm1 = nn.LSTM(
            input_linguistic_dim + acoustic_dim,
            hidden_num,
            num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )  # 入力サイズはここできまる
        self.fc2 = nn.Linear(self.num_direction * hidden_num + 2, z_dim)

        self.use_attention = use_attention
        self.self_attention = MultiHeadAttention(
            n_head, hidden_num*2, hidden_num*2, hidden_num*2, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            hidden_num*2, hidden_num*2, dropout=dropout)
        ##ここまでエンコーダ

        self.fc12 = nn.Linear(
            input_linguistic_dim + z_dim, input_linguistic_dim + z_dim,
        )
        self.lstm2 = nn.LSTM(
            input_linguistic_dim + z_dim,
            hidden_num,
            num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )
        self.fc3 = nn.Linear(self.num_direction * hidden_num, 1)

    def choose_quantized_vector(self, z):  # zはエンコーダの出力
        error = torch.sum((self.quantized_vectors.weight - z) ** 2, dim=1)
        min_index = torch.argmin(error).item()

        return self.quantized_vectors.weight[min_index]

    def quantize_z(self, z_unquantized, epoch):
        if not self.enable_quantize:
            return z_unquantized
        error = torch.sum((z_unquantized.detach().repeat(1, 1, self.num_class).view(-1, self.num_class, self.z_dim) - self.quantized_vectors.weight.detach().view(self.num_class, self.z_dim))**2, dim=2)
        quantized_z_indices = torch.argmin(error, dim=1)
        quantized_z = self.quantized_vectors.weight[quantized_z_indices].view(-1, z_unquantized.size()[1], self.z_dim)
        z = z_unquantized - z_unquantized.detach() + quantized_z
        """
        z = torch.zeros(1, z_unquantized.size(), requires_grad=True).to(device)
        for i in range(z_unquantized.size()[0]):# そろそろこのfor文をなんとかしたい
            z[i] = (
                z_unquantized[i]
                + self.choose_quantized_vector(z_unquantized[i].reshape(-1))
                - z_unquantized[i].detach()
            )
        """

        return z

    def encode(self, linguistic_f, acoustic_f, mora_index):
        x = torch.cat([linguistic_f, acoustic_f], dim=1)
        x = self.fc11(x)
        x = F.relu(x)
        
        out, hc = self.lstm1(torch.cat([x.view(1, x.size()[0], -1)], dim=2))
        mora_index_for_back = np.concatenate([[0], mora_index[:-1] + 1])

        if not self.use_attention:
            out_forward = out[:, :, :hidden_num][:, mora_index]
            out_back = out[:, :, hidden_num:][:, mora_index_for_back]
            out = torch.cat([out_forward, out_back], dim=2)
        else:
            out = self.self_attn_frame2mora(out, mora_index)

        
        h1 = F.relu(out)

        return self.fc2(h1) #ここはモーラ単位しかない

    def init_codebook(self, codebook):
        self.quantized_vectors.weight = codebook

    def self_attn_frame2mora(self, out, mora_index):# 最初と最後のLSTm出力を query, valueにして最後足し算？
        #non_pad_mask = self.get_non_pad_mask(out.view(-1, self.num_direction * hidden_num))

        out_forward = out[:, :, :][:, mora_index]
        mora_index_for_back = np.concatenate([[0], mora_index[:-1] + 1])
        out_backward = out[:, :, :][:, mora_index_for_back]
        #slf_attn_mask = self.get_attn_key_pad_mask(out, out_forward.view(out_forward.size()[1], hidden_num*2))

        mask = torch.ones([1, mora_index.shape[0], out.size()[1]]).to(device).eq(0)
        out_forward, enc_slf_attn_for = self.self_attention(out_forward, out, out, mask=mask)
        out_backward, enc_slf_attn_back = self.self_attention(out_backward, out, out, mask=mask)
        out = out_forward + out_backward
        #out *= non_pad_mask
        out = self.pos_ffn(out)
        #out *= non_pad_mask
        return out

    def get_non_pad_mask(self, seq):
        assert seq.dim() == 2
        return seq.ne(0).type(torch.float).unsqueeze(-1)

    def get_attn_key_pad_mask(self, seq_k, seq_q):
        ''' For masking out the padding part of key sequence. '''

        # Expand to fit the shape of key query attention matrix.
        len_q = seq_q.size(1)
        padding_mask = seq_k.eq(0)
        padding_mask = padding_mask.expand(-1, len_q, -1)  # b x lq x lk


        return padding_mask

    def decode(self, z, linguistic_features, mora_index, tokyo):
        labels = torch.cat([torch.ones([1, linguistic_features.size()[0], 1]), torch.zeros([1, linguistic_features.size()[0], 1])], dim=2) if not tokyo else torch.cat([torch.zeros([1, linguistic_features.size()[0], 1]), torch.ones([1, linguistic_features.size()[0], 1])], dim=2).float().to(device)

        z_tmp = torch.tensor(
            [[0] * self.z_dim] * linguistic_features.size()[0],
            dtype=torch.float32,
            requires_grad=True,
        ).to(device).view(-1, linguistic_features.size()[0], self.z_dim)

        for i, mora_i in enumerate(mora_index):# そろそろこのfor文をなんとかしたい
            prev_index = 0 if i == 0 else int(mora_index[i - 1])
            z_tmp[:, prev_index : int(mora_i)] = z[:,i]

        x = torch.cat(
            [
                linguistic_features,
                z_tmp.view(-1, self.z_dim),
            ],
            dim=1,
        )

        x = self.fc12(x)
        x = F.relu(x)

        h3, (h, c) = self.lstm2(torch.cat([x.view(1, x.size()[0], -1), labels.to(device)], dim=2))
        h3 = F.relu(h3)

        return self.fc3(torch.cat([h3, labels.to(device)], dim=2))  # torch.sigmoid(self.fc3(h3))

    def forward(self, linguistic_features, acoustic_features, mora_index, epoch, tokyo=False):
        z_not_quantized = self.encode(
            linguistic_features, acoustic_features, mora_index, tokyo=tokyo
        )
        z = self.quantize_z(z_not_quantized, epoch)

        return self.decode(z, linguistic_features, mora_index, tokyo=tokyo), z, z_not_quantized


class Rnn(nn.Module):
    def __init__(
        self, bidirectional=True, num_layers=2, accent_label_type=0, output_dim = 1, 
        input_linguistic_dim=289+2
    ):  # accent_label_type: 0;なし, 1; 92次元, 2; H/Lラベル
        super(Rnn, self).__init__()
        self.num_layers = num_layers
        self.num_direction = 2 if bidirectional else 1
        ##ここまでエンコーダ

        if accent_label_type == 0:
            acoustic_linguisic_dim_ = input_linguistic_dim
        elif accent_label_type == 2:
            acoustic_linguisic_dim_ = input_linguistic_dim
        else:
            acoustic_linguisic_dim_ = input_linguistic_dim

        self.lstm2 = nn.LSTM(
            acoustic_linguisic_dim_, 512, num_layers, bidirectional=bidirectional
        )
        self.fc3 = nn.Linear(self.num_direction * 512, output_dim)

    def decode(self, linguistic_features):
        x = linguistic_features.view(linguistic_features.size()[0], 1, -1)
        h3, (h, c) = self.lstm2(x)
        h3 = F.relu(h3)

        return self.fc3(h3)  # torch.sigmoid(self.fc3(h3))

    def forward(self, linguistic_features):

        return self.decode(linguistic_features)



class ALVRnn(nn.Module):
    def __init__(
        self, bidirectional=True, num_layers=2,
        output_dim = 1, input_linguistic_dim=289+2, use_attention = False, n_head=1, dropout=0.2
    ):  # accent_label_type: 0;なし, 1; 92次元, 2; H/Lラベル
        super(ALVRnn, self).__init__()
        self.num_layers = num_layers
        self.num_direction = 2 if bidirectional else 1
        self.use_attention = use_attention
        self.self_attention = MultiHeadAttention(
            n_head, self.num_direction * 512, self.num_direction * 512, self.num_direction * 512, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            self.num_direction * 512, self.num_direction * 512, dropout=dropout)
        ##ここまでエンコーダ

        self.lstm2 = nn.LSTM(
            input_linguistic_dim, 512, num_layers, bidirectional=bidirectional, batch_first=True
        )
        self.fc3 = nn.Linear(self.num_direction * 512, output_dim)

    def decode(self, linguistic_features, mora_index):
        x = linguistic_features.view(1, linguistic_features.size()[0], -1)
        h3, (h, c) = self.lstm2(x)
        h3 = F.relu(h3)

        if self.use_attention:
            h3 = self.self_attn_frame2mora(h3, mora_index)
        else:
            h3 = h3[mora_index]

        return self.fc3(h3)  # torch.sigmoid(self.fc3(h3))

    def forward(self, linguistic_features, mora_index):

        return self.decode(linguistic_features, mora_index)

    def self_attn_frame2mora(self, out, mora_index):# 最初と最後のLSTm出力を query, valueにして最後足し算？
        #non_pad_mask = self.get_non_pad_mask(out.view(-1, self.num_direction * hidden_num))

        out_forward = out[:, :, :][:, mora_index]
        mora_index_for_back = np.concatenate([[0], mora_index[:-1] + 1])
        out_backward = out[:, :, :][:, mora_index_for_back]
        #slf_attn_mask = self.get_attn_key_pad_mask(out, out_forward.view(out_forward.size()[1], hidden_num*2))

        mask = torch.ones([1, mora_index.shape[0], out.size()[1]]).to(device).eq(0)
        out_forward, enc_slf_attn_for = self.self_attention(out_forward, out, out, mask=mask)
        out_backward, enc_slf_attn_back = self.self_attention(out_backward, out, out, mask=mask)
        out = out_forward + out_backward
        #out *= non_pad_mask
        out = self.pos_ffn(out)
        #out *= non_pad_mask
        return out

    def get_non_pad_mask(self, seq):
        assert seq.dim() == 2
        return seq.ne(0).type(torch.float).unsqueeze(-1)

    def get_attn_key_pad_mask(self, seq_k, seq_q):
        ''' For masking out the padding part of key sequence. '''

        # Expand to fit the shape of key query attention matrix.
        len_q = seq_q.size(1)
        padding_mask = seq_k.eq(0)
        padding_mask = padding_mask.expand(-1, len_q, -1)  # b x lq x lk


        return padding_mask


class AccentDiscriminator(nn.Module):
    def __init__(
        self, bidirectional=True, num_layers=2, output_dim_nc=2, output_dim_tf=1, input_linguistic_dim=285+1
    ):  # accent_label_type: 0;なし, 1; 92次元, 2; H/Lラベル
        super(AccentDiscriminator, self).__init__()
        self.num_layers = num_layers
        self.num_direction = 2 if bidirectional else 1
        self.z_dim = 1
        ##ここまでエンコーダ

        self.lstm2 = nn.LSTM(
            input_linguistic_dim, 512, num_layers, bidirectional=bidirectional
        )
        self.fc3 = nn.Linear(self.num_direction * 512, output_dim_nc)
        self.fc4 = nn.Linear(self.num_direction * 512, output_dim_tf)

    def decode(self, linguistic_features, z, mora_index):
        x = linguistic_features.reshape(linguistic_features.size()[0], 1, -1)
        z_tmp = torch.tensor(
            [[0.] * self.z_dim] * linguistic_features.size()[0]
            ).detach().to(device)


        for i, mora_i in enumerate(mora_index):
            prev_index = 0 if i == 0 else int(mora_index[i - 1])
            z_tmp[prev_index : int(mora_i)] = z[i]

        x = torch.cat([
                linguistic_features,
                z_tmp.reshape(-1, self.z_dim),], dim=1,).reshape(linguistic_features.size()[0], 1, -1)


        h3, (h, c) = self.lstm2(x)
        h3 = F.relu(h3[-1])
        h4 = self.fc3(h3)
        h5 = self.fc4(h3)



        return torch.sigmoid(h5.view(-1)), torch.softmax(h4.view(-1, 2), dim=1)

    def forward(self, linguistic_features, z, mora_index):
        linguistic_features_ = linguistic_features[:, :-2]

        return self.decode(linguistic_features_, z, mora_index)



class AccentGenerator(nn.Module):
    def __init__(
        self, bidirectional=True, num_layers=2, output_dim=4, input_linguistic_dim=289+2
    ):  # accent_label_type: 0;なし, 1; 92次元, 2; H/Lラベル
        super(AccentGenerator, self).__init__()
        self.num_layers = num_layers
        self.num_direction = 2 if bidirectional else 1
        ##ここまでエンコーダ

        self.lstm2 = nn.LSTM(
            input_linguistic_dim, 512, num_layers, bidirectional=bidirectional
        )
        self.fc3 = nn.Linear(self.num_direction * 512, output_dim)

    def decode(self, linguistic_features, mora_index):
        x = linguistic_features.view(linguistic_features.size()[0], 1, -1)
        h3, (h, c) = self.lstm2(x)
        h3 = F.relu(h3[mora_index])
        h4 = self.fc3(h3)


        return torch.softmax(h4, dim=2)

    def forward(self, linguistic_features, mora_index):

        return self.decode(linguistic_features, mora_index)


class BinaryFileSource(FileDataSource):
    def __init__(self, data_root, dim, train, valid=True, tokyo=False):
        self.data_root = data_root
        self.dim = dim
        self.train = train
        self.valid = valid
        self.tokyo = tokyo

    def collect_files(self):
        files = sorted(glob(join(self.data_root, "*.bin")))
        # files = files[:len(files)-5] # last 5 is real testset
        train_files = []
        test_files = []
        # train_files, test_files = train_test_split(files, test_size=test_size, random_state=random_state)
        if self.tokyo:
            group_num = 20

            for i, path in enumerate(files):
                if (i - 1) % group_num == 0:  # test
                    if not self.valid:
                        test_files.append(path)
                elif i % group_num == 0:  # valid
                    if self.valid:
                        test_files.append(path)
                else:
                    train_files.append(path)

        else:
            """
            group_num = 13

            for i, path in enumerate(files):
                if (i - 1) % group_num == 0:  # test
                    if not self.valid:
                        test_files.append(path)
                elif i % group_num == 0:  # valid
                    if self.valid:
                        test_files.append(path)
                else:
                    train_files.append(path)
            """
            valid_indices = np.loadtxt('data/valid_indices.csv').astype(int)
            test_indices = np.loadtxt('data/test_indices.csv').astype(int)
            
            for i, path in enumerate(files):
                if i in test_indices:  # test
                    if not self.valid:
                        test_files.append(path)
                elif i in valid_indices:  # valid
                    if self.valid:
                        test_files.append(path)
                else:
                    train_files.append(path)


        if self.train:
            return train_files
        else:
            return test_files

    def collect_features(self, path):
        return np.fromfile(path, dtype=np.float32).reshape(-1, self.dim)


class LBG:
    def __init__(self, num_class=2, z_dim=8):
        self.num_class = num_class
        self.z_dim = z_dim
        self.eps = np.array([1e-2] * z_dim)

    def calc_center(self, x):
        vectors = x.view(-1, self.z_dim)
        center_vec = torch.sum(vectors, dim=0) / vectors.size()[0]

        return center_vec

    def calc_q_vec_init(self, x):
        center_vec = self.calc_center(x).cpu().numpy()
        init_rep_vecs = np.array([center_vec - self.eps, center_vec + self.eps])

        return init_rep_vecs

    def calc_q_vec(self, x):
        # はじめに最初の代表点を求める
        init_rep_vecs = self.calc_q_vec_init(x)
        # K-means で２クラスに分類
        data = x.cpu().numpy()
        kmeans = KMeans(n_clusters=2, init=init_rep_vecs, n_init=1).fit(data)
        rep_vecs = kmeans.cluster_centers_

        for i in range(int(np.log2(self.num_class)) - 1):
            rep_vecs = np.concatenate([rep_vecs + self.eps, rep_vecs - self.eps])
            kmeans = KMeans(n_clusters=2 ** (i + 2), init=rep_vecs).fit(data)
            rep_vecs = kmeans.cluster_centers_

        return rep_vecs
