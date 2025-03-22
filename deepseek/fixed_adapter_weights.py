import torch
import os

# Load your current adapter state dict
adapter_path = "/shared/artifacts/18142399-96ff-4846-b55b-3be3822720f6/checkpoints/AtomicDirectory_checkpoint_56_consolidated_lora/adapter_model.bin"
output_path = "/shared/artifacts/18142399-96ff-4846-b55b-3be3822720f6/checkpoints/AtomicDirectory_checkpoint_56_consolidated_lora_fixed"
os.makedirs(output_path, exist_ok=True)

# Load the state dictionary
state_dict = torch.load(adapter_path)

# Create a new state dict with transformed keys
new_state_dict = {}

for key, value in state_dict.items():
    # 1. Add "base_model." prefix if not present
    if not key.startswith("base_model."):
        new_key = "base_model." + key
    else:
        new_key = key
        
    # 2. Replace ".weight" with ".default.weight"
    if new_key.endswith(".weight"):
        new_key = new_key[:-7] + ".default.weight"
        
    # Store with the new key
    new_state_dict[new_key] = value

# Save the transformed state dict
torch.save(new_state_dict, os.path.join(output_path, "adapter_model.bin"))

# Copy the adapter config if it exists
import shutil
try:
    shutil.copy(os.path.join(os.path.dirname(adapter_path), "adapter_config.json"), 
                os.path.join(output_path, "adapter_config.json"))
    print("Copied adapter_config.json")
except:
    print("No adapter_config.json found, you may need to create one")

print(f"Transformed adapter saved to {output_path}")