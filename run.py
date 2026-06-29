"""
Author Luis Raymundo Pérez Alvarado & Joshua Ian Hernandez Esteves
Description : Script to load the study setup and launch the training
Creation date: 10/09/25
Last update: 14/04/26
Changes history:
                - torch.compilation FAILURE : It was attempted to compile the trainings, but due to
                  due to the variety of graph sizes, it's not supported. Therefore, some functions are noted
                  and some others deactivated
"""
import sys
from utils.Manager import V05

import os
import json

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import torch
#import torch._inductor.config

#torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
#torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None

###
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


try:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    
    new_soft = min(65536, hard)  
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    print(f"✓ File descriptors limit: {new_soft}")
except Exception as e:
    print(f"⚠ No se pudo ajustar file descriptors: {e}")

if torch.cuda.is_available():
        print(f"✓ GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ PyTorch Version: {torch.__version__}")
        print(f"✓ VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print('='*60)
    print('*'*60)
    print("⚠ No se detectó GPU - entrenamiento será efectuado en CPU")
    print('*'*60)
    print('='*60)
    
# Crear y ejecutar manager
print("\n" + "="*60)
print("INICIANDO ENTRENAMIENTO")
print("="*60 + "\n")


# STEP 0 Import data from JSON with experiment data
try:
    json_neuronal_network_data_path = sys.argv[1]
    with open(json_neuronal_network_data_path, 'r') as json_file:
        neuronal_network_experiment_data = json.load(json_file)
except:
    print("Second argument must be a .json file with training configuration data.")
    exit()

# STEP 1 Create a instance for pytorch_geometric_neuronal_network class
pytorch_geometric_nn = V05(neuronal_network_experiment_data)

# STEP 3 Make a training with iterations for each training parameter
pytorch_geometric_nn.main()

print("STUDY HAS BEEN COMPLETED SUCCESFULLY!") 
