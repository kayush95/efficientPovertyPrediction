#!/bin/bash
#SBATCH --partition=atlas --qos=normal
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:2

#SBATCH --job-name="sent_more"
#SBATCH --output=sentinel-%j.out

# only use the following if you want email notification
####SBATCH --mail-user=youremailaddress
####SBATCH --mail-type=ALL

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# # sample process (list hostnames of the nodes you've requested)
# NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
# echo NPROCS=$NPROCS

source /atlas/u/kayush/Single_Shot_Object_Detector/PyTorch-YOLOv3/newPovertyEnv/bin/activate
sh run.sh > cv/normal_more/logs.txt

## SBATCH --nodelist=atlas20
# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

