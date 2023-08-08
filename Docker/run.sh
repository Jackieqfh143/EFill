sudo docker run -it --rm -P \
    --mount type=bind,source=./EFill,target=/home/EFill \  #source地址为本地绝对路径，target为docker内部的相对路径可自行设定
    efill_env:v1

//设置环境变量
export PATH="$PATH:$HOME/miniconda/bin"
