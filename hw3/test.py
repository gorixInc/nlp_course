import torch
device = None
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)
except Exception as e:
    device = e
with open ('test_output.txt', mode='w+') as f:
    f.write(device)