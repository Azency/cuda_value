# .devcontainer/Dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# 基础工具
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake gdb git pkg-config && \
    rm -rf /var/lib/apt/lists/*

# 非 root 用户，避免生成 root 拥有的文件
ARG USERNAME=pengbo
RUN groupadd -g 1011 ${USERNAME} && useradd -s /bin/bash -u 1011-g ${USERNAME} -m ${USERNAME} && usermod -g root ${USERNAME}
USER $USERNAME



WORKDIR /workspace   # Dev Containers 默认把 repo mount 到这里
