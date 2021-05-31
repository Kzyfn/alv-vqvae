# アクセント潜在変数抽出のためのVQ-VAEモデル

アクセント潜在変数抽出のためのVQ-VAEモデル　ソースコードレポジトリです。

以下、学習のための手順を書きます。

## データの準備
参考 nnmnkwii_gallery (リンク)[https://r9y9.github.io/nnmnkwii/latest/nnmnkwii_gallery/notebooks/tts/02-Bidirectional-LSTM%20based%20RNNs%20for%20speech%20synthesis%20using%20OpenJTalk%20%28ja%29.html]

必要なものは
1. 音声データ(wav)
2. HTS方式のラベル
です。

これらから、入力として必要な
1. 音響特徴量ベクトル (メルケプ、F0、非周期性指標)
2. 言語特徴量ベクトル (アクセント情報は入っていない)
3. モーラの区切りのインデックス

を作成します。

### データの設置
音声ファイルを
data/wav
ラベルファイルを
data/label_phone_align
に設置します。
(BASIC5000にしか対応しないようになっています。他のサブセットを使う場合はscp周りのコードを書き換える必要があります。)

具体的には、
util.py create_loader()内 mora_indexを


### 入力情報の作成
#### 特徴量ベクトル
データを設置したら、以下コマンドを実行します。

```python ./scripts/prepare_features.py ./data/ --use_phone_alignment --question_path="./data/questions_jp.hed"```

data/
に X_acoustic、Y_acoustic が作成されていればOKです

#### モーラの区切り
scripts フォルダに移動し、以下コマンドを実行します。

```python mora_index.py```
data/mora_index
に squeezed_ *.csv
が作成されていればOKです

### VQ-VAEモデルの学習

学習は以下コマンドで行います。

```python src/vqvae.py -od [output_dir] -ne [num_epoch] -nl [num_lstm_layers] -zd [z_dim] -mp [preload_model_weight_path]```

