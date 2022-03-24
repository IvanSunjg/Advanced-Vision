#!/bin/bash
fileid="1rYquR66zRKbFYuQPhaM39rMGxjTeBM_-"
# filename="train.zip"
# curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
# curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

gdown --id $fileid
