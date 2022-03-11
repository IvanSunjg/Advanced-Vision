#!/bin/sh
sbatch <<EOT
#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:2
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00
#SBATCH --error="slurm-%j.err"

bash run_experiment.sh "$@"
EOT
