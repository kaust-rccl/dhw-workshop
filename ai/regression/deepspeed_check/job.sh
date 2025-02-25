#!/bin/bash -l

#SBATCH --job-name=bench-tinyIM-ds
#SBATCH --ntasks=2		# Number of CPU processes lanuching work on each GPU (should be equal to the total number of GPUs)
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1      # Number of worker processes for Dataloading feeding to each GPU.
#SBATCH --hint=nomultithread    # Disabling hyperthreading
#SBATCH --time=00:30:00	        # Request for time for the job to run and then terminate automatically (HH:MM:SS)


scontrol show job ${SLURM_JOBID}
module use /sw/modulefiles
module load python
module load openmpi
export TRITON_CACHE_DIR=${PWD}/cache
export MAX_JOBS=${SLURM_CPUS_PER_TASK}
# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
echo "Node IDs of participating nodes ${nodes_array[*]}"

export HOSTFILE=./hostfile
rm $HOSTFILE
for i in ${nodes_array[*]}
do
	echo $i slots=${SLURM_GPUS_PER_NODE} >> ${HOSTFILE}
done

# Get the IP address and set port for MASTER node
head_node="${nodes_array[0]}"
echo "Getting the IP address of the head node ${head_node}"
export master_ip=$(srun -n 1 -N 1 --gpus=1 -w ${head_node} /bin/hostname -i)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "head node is ${master_ip}:${MASTER_PORT}"


echo "Hostname: $(/bin/hostname)"
echo "CPU workers: $workers"

deepspeed --hostfile ./hostfile \
	--launcher SLURM \
	--launcher_args "\--mpi=pmix" \
	--num_nodes=${SLURM_NNODES} \
       	ds_dist.py 
