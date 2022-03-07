# General Setup for MLPcluster

* Install university [vpn](https://computing.help.inf.ed.ac.uk/openvpn)
  * Don't believe this is necessary if you are on `eduroam`
* ssh into your dice `ssh sXXXXXXX@student.ssh.inf.ed.ac.uk`
* ssh into MLP cluster `ssh mlp1`

## Initialize Environment

* Start by downloading the miniconda3 installation file and running it

```sh
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

* Set up terminal initialization. This will automate the conda activation process every time you ssh into the mlp cluster

```sh
cd ~
echo "source .bashrc" >> ".bash_profile"
echo "source activate" >> ".bashrc"
source .bash_profile
```

* Create the conda MLP environment

```sh
conda create -n mlp python=3.8
conda activate mlp
```

## Set Up Environment

* Clone repository

```sh
conda install git
git clone https://github.com/IvanSunjg/Advanced-Vision.git
cd Advanced-Vision/mlp-cluster
```

* Install required packages, e.g. `pytorch` and `gdown`

```sh
bash install/install.sh
```

* Download datasets from google drive

```sh
bash install/download_datasets.sh
```

## Submit job

* Make sure you are inside the `mlp-cluster` directory
* Add the execution of your script to the bottom of `send_model_to_job.sh`. Nothing else in the script needs to be changed
* Run `sbatch send_model_to_job.sh` to submit a job to the cluster
* Now you will see a job id and you can view the terminal output will be output into `slurm-{job id}.out` in your current directory
  * You can view the job queue using `squeue` or `squeue -u sXXXXXXX` or `squeue -j {job id}`
* You can modify the number of GPUs and other parameters in `send_model_to_job.sh`

## Additional Information

* More information can be accessed at the original MLP cluster set up [guide](https://github.com/VICO-UoE/mlpractical/blob/mlp2021-22/mlp_cluster_tutorial/mlp_cluster_quick_start_up.md)
* The Slurm [cheat sheet](https://github.com/JIC-CSB/SLURM-cheat-sheet) and [documentation](https://slurm.schedmd.com/) can be referenced for working more closely with the cluster manager.
