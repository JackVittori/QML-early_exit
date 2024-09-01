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

num_qubits = 8
num_layers = 8
first_pair = [0,1]
second_pair = [2,3]

def calculate_mcm_accuracy(dictionary: Dict, report: bool = False):
    early_correct = sum([1 for label, prediction in dictionary['early'] if label == prediction])
    early_accuracy = early_correct/len(dictionary['early'])
    final_correct = sum([1 for label, prediction in dictionary['final'] if label == prediction])
    final_accuracy = final_correct/len(dictionary['final'])
    if report:
        print(early_correct, "elements have been correctly classified in the early exit over", len(dictionary['early']), "that have been early classified, with an accuracy of: ", early_accuracy, "\n", final_correct, "elements have been correctly classified in the final evaluation over", len(dictionary['final']), "that have not been early classified, with an accuracy of: ", final_accuracy)
    return early_accuracy, final_accuracy

def early_evaluation(parameters: Dict, state: torch.Tensor = None):
    mcms = []
    if state is not None:
        # state vector initialization with input
        qml.QubitStateVector(state, wires=range(num_qubits))

    for i in range(4):
        for j in range(num_qubits):
            qml.RX(parameters[f'layer_{i}'][j, 0], wires=j)
            qml.RY(parameters[f'layer_{i}'][j, 1], wires=j)
            qml.RZ(parameters[f'layer_{i}'][j, 2], wires=j)
        for j in range(num_qubits):
            qml.CNOT(wires=[j, (j + 1) % num_qubits])
    for w in first_pair:
        mcms.append(qml.measure(wires=w))
    return mcms


dev = qml.device("default.qubit", shots=20)


@qml.qnode(dev)
def circuit_early_evaluation(parameters: Dict, state: torch.Tensor = None):
    results = early_evaluation(parameters, state)
    return qml.probs(op=results)


def full_evaluation(parameters: Dict, state: torch.Tensor = None):
    mcms = []
    if state is not None:
        # state vector initialization with input
        qml.QubitStateVector(state, wires=range(num_qubits))

    for i in range(4):
        for j in range(num_qubits):
            qml.RX(parameters[f'layer_{i}'][j, 0], wires=j)
            qml.RY(parameters[f'layer_{i}'][j, 1], wires=j)
            qml.RZ(parameters[f'layer_{i}'][j, 2], wires=j)
        for j in range(num_qubits):
            qml.CNOT(wires=[j, (j + 1) % num_qubits])
    for w in first_pair:
        mcms.append(qml.measure(wires=w))

    for i in range(4, num_layers):
        for j in range(num_qubits):
            qml.RX(parameters[f'layer_{i}'][j, 0], wires=j)
            qml.RY(parameters[f'layer_{i}'][j, 1], wires=j)
            qml.RZ(parameters[f'layer_{i}'][j, 2], wires=j)
        for j in range(num_qubits):
            qml.CNOT(wires=[j, (j + 1) % num_qubits])

    for w in second_pair:
        mcms.append(qml.measure(wires=w))
    return mcms


dev = qml.device("default.qubit", shots=20)


@qml.qnode(dev)
def full_evaluation_circuit(parameters: Dict, state: torch.Tensor = None):
    results = full_evaluation(parameters, state)
    mcm = results[0:2]
    fm = results[2:]
    return qml.probs(op=fm)

