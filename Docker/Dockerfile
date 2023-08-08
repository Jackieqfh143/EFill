FROM ubuntu:18.04
WORKDIR /home
RUN set -x; buildDeps='git wget openssl libssl-dev zip g++ vim' \
    && apt update && apt upgrade \
    && apt-get install -y $buildDeps \

RUN export PATH="$PATH:$HOME/miniconda/bin" \
	&& pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu \
	&& python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
  	
