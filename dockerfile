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

# 3) python コマンドを python3 にリンクする（必要であれば）
RUN ln -s /usr/bin/python3 /usr/bin/python

# 4) 必要なライブラリもインストール
RUN pip install torch transformers peft accelerate bitsandbytes datasets pandas

# スクリプトをコピー
COPY train.py /workspace/train.py

CMD ["/bin/bash"]
