# General setup for MLPcluster
* install uni vpn
* follow tutorial to access cluster and setup MLP cluster environment and read instructions what is slurm
## Download dataset and missing package from mlp
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
conda activate mlp
conda install scikit-learn
```
## Submit job

```
sbatch model.sh

```

