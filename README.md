## 作業用紗夜さんボイス

バンドリの会話シーンからキャラごとの発言シーンを抽出・結合して動画ファイルと音声ファイルとして出力します。

[生成例 (Dropbox)](https://www.dropbox.com/s/rfoxj7whwdgs6rq/0d9dee4f689e1ea10209d614fd74af77.mp4?dl=0)

## 必要なもの

- Anaconda3
  - Python 3.8
- CUDA Toolkit 11.1
- cuDNN SDK 8
- NVIDIA GPU Driver 460.x
  - detail: [GPU Support  \|  TensorFlow](https://www.tensorflow.org/install/gpu?hl=ja#software_requirements)

## セットアップ

```sh
conda env create -f=env_name.yml
activate py38
```

## 使い方

### 既存のモデルを使用する場合

```sh
python src/initialize.py
python src/create_record.py
python src/create_movie.py
python src/create_product.py
```

### 学習モデルを再構築する場合

```sh
python src/initialize.py
python src/create_dataset.py
python src/create_model.py
python src/create_record.py
python src/create_movie.py
python src/create_product.py
```
