# Use the official PostgreSQL image as the base image
FROM postgres

# Install the pgvector extension
RUN apt-get update && apt-get install -y git build-essential clang llvm-dev
RUN git clone --branch v0.4.4 https://github.com/pgvector/pgvector.git
RUN cd pgvector && make clean && make && make install

# Install the NVIDIA PyTorch container
RUN apt-get install -y python3-pip
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt