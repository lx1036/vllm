


from safetensors.torch import load_file
from fastsafetensors import SafeTensorsFileLoader, SingleGroup


def main():


    file = "a.safetensors"
    original_keys = load_file(file)
    print(original_keys)
    """
    {'a0': tensor([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
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
        [15., 15., 15., 15., 15., 15., 15., 15.]], dtype=torch.float16)}
    """


    device = "cuda:0"
    loader = SafeTensorsFileLoader(pg=SingleGroup(), device=device, nogds=False, debug_log=True)
    loader.add_filenames({0: [file]})
    files_buffer = loader.copy_files_to_device()
    for key in loader.get_keys():
        print(f"{key} {loader.get_shape(key)} {loader.frames[key].data_offsets} {files_buffer.get_tensor(key).dtype} {original_keys[key].dtype}")

    """
    add_filenames 1: path=a.safetensors
[DEBUG] raw_gds_file_handle: fd=79, cf_handle=0xabababab0000004f, elapsed=3807 us
[DEBUG] gds_device_buffer.cufile_register: addr=0x7f0e09e00000, offset=0, length=4096, register=2877 us
submit_io: new buf, addr=0x7f0e09e00000
[DEBUG] gds_file_reader._thread: cuFileRead(fh, 0x7f0e09e00000, length=4096, off=0, ptr_off=0, count=0)=360
[DEBUG] gds_file_reader._thread: fh=0xabababab0000004f, offset=0, length=4096, count=360, read=181 us, notify=1 us
[DEBUG] gds_device_buffer.cufile_deregister: addr=0x7f0e09e00000, offset=0, elapsed=651 us
[DEBUG] ~raw_gds_file_handle: cuFileHandleDeregister: cf_handle=0xabababab0000004f
[DEBUG] ~raw_gds_file_handle: close: fd=79
wait_io: fix misalignment, src=0x7f0e09e00000, misaligned_bytes=8, count=0, tmp=0x7f0da0000000
[DEBUG] gds_device_buffer.memmove: dst=0x7f0e09e00000, src=0x7f0e09e00008, tmp=0x7f0da0000000, length=4088, elapsed=68 us
wait_io: tensor=a0
a0 [16, 8] [0, 256] torch.float16 torch.float16
    """

if __name__ == "__main__":
    main()





