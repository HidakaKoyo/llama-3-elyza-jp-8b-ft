FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 1) システムパッケージを更新し、Python3 & pip をインストール
RUN apt-get update && apt-get install -y \
  python3 \
  python3-distutils \
  curl \
  && rm -rf /var/lib/apt/lists/*

# 2) pip のインストール
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py

# 3) python コマンドを python3 にリンク
RUN ln -s /usr/bin/python3 /usr/bin/python

# 4) 必要なライブラリをインストール
RUN pip install torch transformers peft accelerate bitsandbytes datasets pandas pynvml

# 5) ソースコードをコピー
COPY . /app

# 6) 作業ディレクトリを /app に設定
WORKDIR /app

# CMD でデフォルトの実行コマンドを指定しておく例
# CMD ["python", "workspace/train.py", "--output_dir", "/opt/artifact", "--num_train_epochs", "2", "--batch_size", "2"]
