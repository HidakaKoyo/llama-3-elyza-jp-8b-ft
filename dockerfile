# (例) nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 をベースにする
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 必要ライブラリのインストール
RUN apt-get update && apt-get install -y \
  git \
  wget \
  curl \
  python3.9 \
  python3.9-distutils \
  vim \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# pip インストール
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.9 get-pip.py

# 必要 Python パッケージのインストール
RUN pip install --no-cache-dir \
  torch \
  transformers \
  datasets \
  peft \
  accelerate \
  bitsandbytes \
  sentencepiece \
  # 以下は便利ツール
  pandas \
  jupyter

# コンテナ内の作業ディレクトリ
WORKDIR /workspace

# スクリプトやデータを追加
# ADD ./train.py /workspace

CMD ["/bin/bash"]
