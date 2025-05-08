import torch
import torch.nn as nn
import GPUtil

def print_gpu_memory():
    """Print detailed GPU memory usage."""
    gpu = GPUtil.getGPUs()[0]
    print(f"\nGPU Memory Usage:")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"GPU Memory Used: {gpu.memoryUsed} MB")
    print(f"GPU Memory Free: {gpu.memoryFree} MB")
    print(f"GPU Memory Total: {gpu.memoryTotal} MB")
    print(f"GPU Utilization: {gpu.load * 100:.1f}%")

def verify_gradients(model: nn.Module) -> bool:
    """Verify that model parameters have gradients enabled."""
    has_gradients = False
    total_params = 0
    trainable_params = 0
    
    print("\nModel Parameter Details:")
    print("-" * 50)
    
    # Check each module separately
    for name, module in model.named_children():
        module_params = 0
        module_trainable = 0
        for param in module.parameters():
            module_params += 1
            if param.requires_grad:
                module_trainable += 1
                has_gradients = True
        
        total_params += module_params
        trainable_params += module_trainable
        
        print(f"{name}:")
        print(f"  Parameters: {module_params}")
        if module_params > 0:
            print(f"  Trainable: {module_trainable} ({module_trainable/module_params*100:.1f}%)")
        else:
            print("  Trainable: 0 (0.0%)")
    
    print("-" * 50)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params} ({trainable_params/total_params*100:.1f}%)")
    
    if not has_gradients:
        print("WARNING: No parameters have gradients enabled!")
    
    return has_gradients 