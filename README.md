# General setup for MLPcluster
* Install university vpn[https://computing.help.inf.ed.ac.uk/openvpn]
* ssh into your dice ```ssh sxxxxxx@student.ssh.inf.ed.ac.uk```
* ssh into MLP cluster ```ssh mlp1```
## Setup environment
* Start by downloading the miniconda3 installation file using ```wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh```
* Now run the installation using ```bash Miniconda3-latest-Linux-x86_64.sh```
* Activate environment ```source activate``` 
* Create environment  ```conda create -n mlp python=3.8```
* Run ```source activate mlp``` then ```conda activate mlp```
* Install git ```conda install git```
* Config git ```git config --global user.name "[your name]"; git config --global user.email "[matric-number]@sms.ed.ac.uk"```
* Clone our repo ```git clone https://github.com/IvanSunjg/Advanced-Vision.git```
* cd ```Advanced-Vision```
* Checkout MLPcluster branch ```git checkout MLPcluster```
* Run ```bash install.sh``` to install required packages
## Download dataset and resnet50 checkpoint
* Run ```bash download.sh```
* This will download everything from my google drive unzip it and then delete zip file
## Submit job
* Script takes care of path so don't need to modify anything
* Run ```sbatch send_model_to_job.sh```
* Now you will see a job id then ```nano "id".out``` to see the output
* We use 2 GPUs as using only one can run out of memory. But you can modify this in send_model_to_job.sh
