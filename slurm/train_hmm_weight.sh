#! /bin/bash

#================= Part 1 : job parameters ============
#SBATCH --partition=gpu
#SBATCH --account=cmsgpu
#SBATCH --qos=cmsnormal
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16GB
#SBATCH --job-name=weaver-train
#SBATCH --gres=gpu:v100:2
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4

#============== Part 2 : job workload ===================

# source your environment file if exists
# replace /path/to/your/env_file with your real env file path
source /afs/ihep.ac.cn/users/k/kouhao/conda.sh

# replace /path/to/your/mpi_program with your real MPI program path
cd /publicfs/cms/user/kouhao/weaver-core-dev/weaver
./scripts/train_hmm_vbf_weight.sh run 0,1
