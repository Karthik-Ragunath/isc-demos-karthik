import torch
import pprint

# Path to your adapter_model.bin file
adapter_path = "/shared/artifacts/consolidated_checkpoint/adapter_model.bin"

# Load the state dictionary
state_dict = torch.load(adapter_path)

# Print the keys
print("Number of keys:", len(state_dict.keys()))
print("\Print all keys:")
pprint.pprint(list(state_dict.keys()))

# Optional: Print all keys (can be very long)
# pprint.pprint(list(state_dict.keys()))

# Analyze key patterns to understand the structure
key_prefixes = set()
for key in state_dict.keys():
    # Extract prefix pattern (e.g., up to the first few components)
    parts = key.split('.')
    if len(parts) >= 3:
        prefix = '.'.join(parts[:3])
        key_prefixes.add(prefix)

print("\nUnique key prefixes in the state dict:")
pprint.pprint(sorted(list(key_prefixes)))

# Count parameters
total_params = sum(p.numel() for p in state_dict.values())
print(f"\nTotal number of parameters: {total_params:,}")