## Convolutional VAE

### 概要

畳み込み変分オートエンコーダをpytorchを用いて実装した.

モデルテストには, mnistデータを使用.

参考文献）[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

### 実装内容

- `/notebooks/ConvVAE.ipynb`
- `/src/functions`
- `/src/models`

### 結果

Loss

![losses](./results/losses.png)

入力画像

![input](./results/input.png)

エンコード後の入力画像（潜在変数）

![input_encoded](./results/input_encoded.png)

出力画像

![output](./results/output.png)

ネットワーク図

![model](./model.png)