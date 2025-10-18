
import torch.distributed as dist
import torch
from fastsafetensors import SafeTensorsFileLoader


# terminal1
# FASTSAFETENSORS_ENABLE_INIT_LOG=1 torchrun --nnodes=2 --master_addr=0.0.0.0 --master_port=1234 --node_rank=0 run_parallel.py

# terminal2
# FASTSAFETENSORS_ENABLE_INIT_LOG=1 torchrun --nnodes=2 --master_addr=0.0.0.0 --master_port=1234 --node_rank=1 run_parallel.py


def main():

    dist.init_process_group(backend="gloo")
    dist.barrier()
    process_group = dist.group.WORLD
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    loader = SafeTensorsFileLoader(pg=process_group, device=device, nogds=True, debug_log=True) # nogds=False
    loader.add_filenames({0: ["a.safetensors"], 1: ["b.safetensors"]})

    # load a.safetensors to rank 0 GPU and b.safetensors to rank 1 GPU
    files_buffer = loader.copy_files_to_device()

    # every rank must call get_tensor and get_sharded in the same order since they internally call torch.distributed collective ops
    tensor_a0 = files_buffer.get_tensor(tensor_name="a0") # broadcast
    tensor_b0_sharded = files_buffer.get_sharded(tensor_name="b0", dim=1) # partition and scatter

    print(f"RANK {process_group.rank()}: tensor_a0={tensor_a0}")
    print(f"RANK {process_group.rank()}: tensor_b0_sharded={tensor_b0_sharded}")
    files_buffer.close()
    loader.close()


"""
nogds=True

add_filenames 2: path=b.safetensors
[DEBUG] nogds_file_reader.submit_read: cudaHostAlloc, addr=0x7fab46800000, size=16777216, elapsed=8182 us
[DEBUG] nogds_file_reader.submit_read #3, thread_id=1
submit_io: new buf, addr=0x7fab27e00000
[DEBUG] nogds_file_reader._thread: read (mmap=0), fd=72, offset=104, count=256, c=256, copy=5 us, cuda_copy=254 us
wait_io: tensor=b0
shuffle: broadcast, tensor_name=a0, shape=[16, 8], self.rank=0, pg.rank()=1, has_tensor=False
shuffle: scatter, tensor_name=b0, shape=[16, 8]->[16, 4], self.rank=1, pg.rank()=1, rank_slices=[(slice(None, None, None), slice(0, 4, 1)), (slice(None, None, None), slice(4, 8, 1))], len(scatter_list)=2
_get_tensor: free_dev_ptrs, lidx=0, src=b.safetensors
free_dev_ptrs: delete buf, addr=0x7fab27e00000
RANK 1: tensor_a0=tensor([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.],
        [ 3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.],
        [ 4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.],
        [ 5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.],
        [ 6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.],
        [ 7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.],
        [ 8.,  8.,  8.,  8.,  8.,  8.,  8.,  8.],
        [ 9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.],
        [10., 10., 10., 10., 10., 10., 10., 10.],
        [11., 11., 11., 11., 11., 11., 11., 11.],
        [12., 12., 12., 12., 12., 12., 12., 12.],
        [13., 13., 13., 13., 13., 13., 13., 13.],
        [14., 14., 14., 14., 14., 14., 14., 14.],
        [15., 15., 15., 15., 15., 15., 15., 15.]], device='cuda:0',
       dtype=torch.float16)
RANK 1: tensor_b0_sharded=tensor([[ 0.,  0.,  0.,  0.],
        [ 1.,  1.,  1.,  1.],
        [ 2.,  2.,  2.,  2.],
        [ 3.,  3.,  3.,  3.],
        [ 4.,  4.,  4.,  4.],
        [ 5.,  5.,  5.,  5.],
        [ 6.,  6.,  6.,  6.],
        [ 7.,  7.,  7.,  7.],
        [ 8.,  8.,  8.,  8.],
        [ 9.,  9.,  9.,  9.],
        [10., 10., 10., 10.],
        [11., 11., 11., 11.],
        [12., 12., 12., 12.],
        [13., 13., 13., 13.],
        [14., 14., 14., 14.],
        [15., 15., 15., 15.]], device='cuda:0', dtype=torch.float16)
[DEBUG] cudaFreeHost, addr=0x7fab46800000, size=16777216
[DEBUG] ~nogds_file_reader: elapsed=3406 us



nogds=False

[DEBUG] loaded: libnuma.so.1
[DEBUG] device count=2, cuda_found=1
[DEBUG] loaded: libcudart.so
[DEBUG] loaded: libcufile.so.0 (ver: 1.15.0)
add_filenames 2: path=b.safetensors
[DEBUG] raw_gds_file_handle: fd=72, cf_handle=0xabababab00000048, elapsed=5768 us
[DEBUG] gds_device_buffer.cufile_register: addr=0x7fd99be00000, offset=0, length=4096, register=3431 us
submit_io: new buf, addr=0x7fd99be00000
[DEBUG] gds_file_reader._thread: cuFileRead(fh, 0x7fd99be00000, length=4096, off=0, ptr_off=0, count=0)=360
[DEBUG] gds_file_reader._thread: fh=0xabababab00000048, offset=0, length=4096, count=360, read=303 us, notify=1 us
[DEBUG] gds_device_buffer.cufile_deregister: addr=0x7fd99be00000, offset=0, elapsed=907 us
[DEBUG] ~raw_gds_file_handle: cuFileHandleDeregister: cf_handle=0xabababab00000048
[DEBUG] ~raw_gds_file_handle: close: fd=72
wait_io: fix misalignment, src=0x7fd99be00000, misaligned_bytes=8, count=0, tmp=0x7fd920000000
[DEBUG] gds_device_buffer.memmove: dst=0x7fd99be00000, src=0x7fd99be00008, tmp=0x7fd920000000, length=4088, elapsed=61 us
wait_io: tensor=b0
shuffle: broadcast, tensor_name=a0, shape=[16, 8], self.rank=0, pg.rank()=1, has_tensor=False
shuffle: scatter, tensor_name=b0, shape=[16, 8]->[16, 4], self.rank=1, pg.rank()=1, rank_slices=[(slice(None, None, None), slice(0, 4, 1)), (slice(None, None, None), slice(4, 8, 1))], len(scatter_list)=2
_get_tensor: free_dev_ptrs, lidx=0, src=b.safetensors
free_dev_ptrs: delete buf, addr=0x7fd99be00000
RANK 1: tensor_a0=tensor([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.],
        [ 3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.],
        [ 4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.],
        [ 5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.],
        [ 6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.],
        [ 7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.],
        [ 8.,  8.,  8.,  8.,  8.,  8.,  8.,  8.],
        [ 9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.],
        [10., 10., 10., 10., 10., 10., 10., 10.],
        [11., 11., 11., 11., 11., 11., 11., 11.],
        [12., 12., 12., 12., 12., 12., 12., 12.],
        [13., 13., 13., 13., 13., 13., 13., 13.],
        [14., 14., 14., 14., 14., 14., 14., 14.],
        [15., 15., 15., 15., 15., 15., 15., 15.]], device='cuda:0',
       dtype=torch.float16)
RANK 1: tensor_b0_sharded=tensor([[ 0.,  0.,  0.,  0.],
        [ 1.,  1.,  1.,  1.],
        [ 2.,  2.,  2.,  2.],
        [ 3.,  3.,  3.,  3.],
        [ 4.,  4.,  4.,  4.],
        [ 5.,  5.,  5.,  5.],
        [ 6.,  6.,  6.,  6.],
        [ 7.,  7.,  7.,  7.],
        [ 8.,  8.,  8.,  8.],
        [ 9.,  9.,  9.,  9.],
        [10., 10., 10., 10.],
        [11., 11., 11., 11.],
        [12., 12., 12., 12.],
        [13., 13., 13., 13.],
        [14., 14., 14., 14.],
        [15., 15., 15., 15.]], device='cuda:0', dtype=torch.float16)


"""

if __name__ == "__main__":
    main()
