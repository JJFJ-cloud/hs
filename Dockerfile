# 使用 Conda 基础镜像
FROM continuumio/miniconda3:latest

# 设置工作目录
WORKDIR /app

# 复制 environment.yml 文件
COPY environment.yml .

# 创建 Conda 环境
RUN conda env create -f environment.yml

# 激活 Conda 环境
RUN echo "source activate $(head -1 environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 environment.yml | cut -d' ' -f2)/bin:$PATH

# 复制应用代码
COPY . .

# 运行应用
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]