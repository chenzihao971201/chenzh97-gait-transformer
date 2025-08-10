# ===== Base: CUDA runtime + cuDNN + Ubuntu =====
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 基础依赖 + Python
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev git build-essential \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装 PyTorch（与 CUDA 12.1 匹配）
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
      --index-url https://download.pytorch.org/whl/cu121

# 再装其余依赖（来自你的 requirements.txt）
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝项目代码（.dockerignore 会排除 data/、runs/ 等）
COPY . /app

# 直接运行训练脚本（GPU 信息打印放到 train.py 里）
CMD ["python", "train.py"]