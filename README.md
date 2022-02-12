# General setup for MLPcluster
* install uni vpn
* follow tutorial to access cluster and setup MLP cluster environment and read instructions what is slurm
## Download dataset
```
cd
mkdir AVcw
cd AVcw
chmod +x dw_trainzip.sh 
chmod +x dw_resnet.sh
./dw_trainzip.sh 
./dw_resnet.sh
! unzip train.zip
source .bashrc
conda install scikit-learn
```
## Unzip train.zip to colab
* In a separate cell insert the code below and run
```
!cd /content/data/
! unzip /content/gdrive/MyDrive/train.zip
```
## Start training
* In a separate cell copy baseline_model_colab.py code and run
