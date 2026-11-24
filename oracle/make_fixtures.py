import torch
from oracle import torch_sdpa
import numpy as np
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def make_folder_and_file(file_str):
    file_paths = []
    folder_path = (f"{BASE_DIR}/fixtures/{file_str}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    ary = ['q', 'K', 'V', 'expected']

    for i in ary:

        temp_path = (f"{BASE_DIR}/fixtures/{file_str}/{i}.bin")
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        file_paths.append(temp_path)

    return file_paths


def get_tensors():

    lengths_dims = [1, 100, 1024, 4097, 8192]
    H, D = 32, 128
    for length_L in lengths_dims:

        file_str = 'len' + '_' + str(length_L)
        file_paths = make_folder_and_file(file_str=file_str)

        for i in file_paths:
            print(i)

        q = torch.randn(H, D, dtype=torch.float16)
        K = torch.randn(H, length_L, D, dtype=torch.float16)
        V = torch.randn(H, length_L, D, dtype=torch.float16)

        expected = torch_sdpa(q, K, V)
        expected_numpy = expected.cpu().numpy()

        

get_tensors()