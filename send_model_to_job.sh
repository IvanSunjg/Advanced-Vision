#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:2
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/
# Activate the relevant virtual environment:


source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

python model.py --path /home/${STUDENT_ID}/Advanced-Vision --num_epochs 15 --optimizer_type 'SGD' --lr 0.006 --weight_decay 0.006 --momentum 0.5 --beta1 0 --beta2 0 --amsgrad False --loss_function 'CEL' --reduction 'mean' --label_smoothing 0.0 --step_size 7 --gamma 0.1 --num_of_frozen_blocks 2 --argument1 3 --argument2 50 

python model.py --path /home/${STUDENT_ID}/Advanced-Vision --num_epochs 15 --optimizer_type 'Adam' --lr 0.006 --weight_decay 0.006 --momentum 0 --beta1 0.9 --beta2 0.999 --amsgrad False --loss_function 'KLD' --reduction 'mean' --label_smoothing 0.0 --step_size 7 --gamma 0.1 --num_of_frozen_blocks 2 --argument1 0 --argument2 0 

