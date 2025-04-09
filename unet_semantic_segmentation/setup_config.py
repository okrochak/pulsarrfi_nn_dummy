import os
directories = [
    "./syn_data",
    "./syn_data/runtime/",
    "./syn_data/payloads/",
    "./syn_data/model/",    
]
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"MESSAGE: Created directory: {directory}")
    else:
        print(f"MESSAGE: Directory already exists: {directory}")

print("MESSAGE: All necessary directories are verified.")