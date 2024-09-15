import torch
import pennylane as qml
from pennylane import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from data_utils import mnist_preparation
from tqdm import tqdm
import matplotlib as plt
from OriginalModel import FullQuantumModel, QuantumCircuit
from pennylane import Device
from pennylane.measurements import StateMP
from torch.nn import Module, ParameterDict
import matplotlib.pyplot as plt
import warnings
from typing import Optional, Dict, List, Any
from torch.utils.data import DataLoader, dataloader
from time import time
import math
from pennylane.measurements import MidMeasureMP
from evaluationUtils import early_evaluation, circuit_early_evaluation, full_evaluation, full_evaluation_circuit

"""
MCMCircuit class defining quantum computation integrating Pennylane with Pytorch and containing quantum circuit logic.
"""
params: ParameterDict
dev: Device
num_qubits: int
num_layers: int
early_exits: List[int] #early-exit indexes

def __init__(self, num_qubits: int, num_layers: int, num_shots: int, early_exits = List[int]):
    super().__init__()
    self.num_qubits = num_qubits
    self.num_layers = num_layers
    self.early_exits = early_exits
    self.params = torch.nn.ParameterDict({
        f'layer_{i}': torch.nn.Parameter(torch.rand(num_qubits, 3, requires_grad=True))
        for i in range(num_layers)
    })

    for num in self.early_exits:
        if num >= self.num_layers:
            raise ValueError(f"Error: The early exit {num} is out of number of layers given, that is from 0 to {self.num_layers-1}")


    def