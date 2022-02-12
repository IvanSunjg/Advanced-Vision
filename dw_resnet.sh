#!/bin/bash
fileid="13QJMnI_sQwyFpNOZLRjKZMqzdpnm9HrG"
filename="resnet50_fconv_model_best.pth.tar"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

