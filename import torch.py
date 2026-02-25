import torch

# Load the .pth file
checkpoint = torch.load('model.pth', map_location='cpu')

# Inspect what's inside
if isinstance(checkpoint, dict):
    print("Keys:", checkpoint.keys())
    
    # Common keys in checkpoints
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint  # Might be a raw state_dict
    
    # Print layer names and shapes
    for name, param in state_dict.items():
        print(f"{name}: {param.shape}")
else:
    # It's likely a raw state_dict
    for name, param in checkpoint.items():
        print(f"{name}: {param.shape}")