import deepspeed
import torch
import os

def test_distributed():
    """Tests deepspeed.init_distributed() and basic distributed communication."""

    # 1. Initialize distributed environment
    deepspeed.init_distributed()  # Or "nccl" for GPUs

    # 2. Get rank and world size
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # 3. Print information about the distributed setup
    print(f"Rank {rank} of {world_size} initialized.")

    # 4. Basic communication test: send a tensor from rank 0 to all other ranks
    if rank == 0:
        tensor_to_send = torch.tensor([rank + 1], dtype=torch.int32)  # Example tensor
        for dest_rank in range(1, world_size):
            torch.distributed.send(tensor=tensor_to_send, dst=dest_rank)
        print(f"Rank 0 sent tensor {tensor_to_send} to other ranks.")
    else:
        received_tensor = torch.zeros(1, dtype=torch.int32)
        torch.distributed.recv(tensor=received_tensor, src=0)
        print(f"Rank {rank} received tensor {received_tensor} from rank 0.")

    # 5. Collective communication test: broadcast a tensor
    tensor_to_broadcast = torch.tensor([rank * 10], dtype=torch.int32)
    torch.distributed.broadcast(tensor=tensor_to_broadcast, src=0)
    print(f"Rank {rank} has tensor after broadcast: {tensor_to_broadcast}")

    # 6. Gather test: gather tensors from all ranks to rank 0
    tensor_to_gather = torch.tensor([rank * 100], dtype=torch.int32)
    if rank == 0:
      gathered_tensors = [torch.zeros(1, dtype=torch.int32) for _ in range(world_size)]
      torch.distributed.gather(tensor=tensor_to_gather, gather_list=gathered_tensors, dst=0)
      print(f"Rank 0 gathered tensors: {gathered_tensors}")
    else:
      torch.distributed.gather(tensor=tensor_to_gather, dst=0)

    # 7. All reduce test: sum tensors on all ranks
    tensor_to_reduce = torch.tensor([rank * 1000], dtype=torch.int32)
    torch.distributed.all_reduce(tensor=tensor_to_reduce, op=torch.distributed.ReduceOp.SUM)
    print(f"Rank {rank} has tensor after all_reduce: {tensor_to_reduce}")

    # 8. Barrier: wait for all processes to finish
    torch.distributed.barrier()
    print(f"Rank {rank} reached the barrier.")

    # 9. Destroy the process group (important cleanup)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    test_distributed()
