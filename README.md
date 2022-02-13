# General setup for MLPcluster
* Install university vpn[https://computing.help.inf.ed.ac.uk/openvpn]
* ssh into your dice ssh ```sxxxxxx@student.ssh.inf.ed.ac.uk```
* ssh into MLP cluster ```ssh mlp1```
## Setup environment
* Start by downloading the miniconda3 installation file using ```wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh```
* Now run the installation using ```bash Miniconda3-latest-Linux-x86_64.sh```
* Activate environment ```source activate``` 
* Create environment  ```conda create -n mlp python=3.8```
* Run ```source activate mlp``` then ```conda activate mlp```
* Install git ```conda install git```
* Config git ```git config --global user.name "[your name]"; git config --global user.email "[matric-number]@sms.ed.ac.uk"```
* Clone our repo 
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

