import numba
from numba import cuda
from numba.cuda.cudadrv import enums

device = cuda.select_device(0)
print(device.name)

attribs= [name.replace("CU_DEVICE_ATTRIBUTE_", "") for name in dir(enums) if name.startswith("CU_DEVICE_ATTRIBUTE_")]
for attr in attribs:
    
    if attr == "CLOCK_RATE" or attr == "MULTIPROCESSOR_COUNT" or attr == "COMPUTE_CAPABILITY_MAJOR" or attr == "COMPUTE_CAPABILITY_MINOR":
        print(attr, '=', getattr(device, attr))        
device_memory =device.get_primary_context().get_memory_info()

print(device_memory)