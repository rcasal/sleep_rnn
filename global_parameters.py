
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cuda:1'
dtype_data = torch.float32
dtype_target = torch.int64