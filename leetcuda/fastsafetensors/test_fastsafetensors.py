from fastsafetensors import SafeTensorsMetadata
from fastsafetensors.copier.gds import GdsFileCopier
from fastsafetensors.frameworks import get_framework_op
from fastsafetensors import cpp
from fastsafetensors.st_types import Device


def test_GdsFileCopier():
    file = "a.safetensors"
    framework = get_framework_op("torch")
    metadata = SafeTensorsMetadata.from_file(filename=file, framework=framework)
    print(metadata)
    '''
    {'__metadata__': OrderedDict([('fst', 'sample')]), 'tensors': {'a0': {'dtype': <DType.F16: 'F16'>, 'shape': [16, 8], 'data_offsets': [0, 256]}}}
    '''

    use_cuda = True
    device = "cuda:0"
    reader = cpp.gds_file_reader(4, use_cuda)
    gds_copier = GdsFileCopier(metadata, Device.from_str(device), reader, framework, True)
    gds_device_buf = gds_copier.submit_io(True, 10 * 1024 * 1024 * 1024)
    tensors = gds_copier.wait_io(gds_device_buf)




