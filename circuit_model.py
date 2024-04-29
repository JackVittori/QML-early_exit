import torch
import pennylane as qml
from torch.autograd import Variable
class BasicLayer:
    def __init__(self, n_qubits: int, params = None):
        self.n_qubits = n_qubits
        if params is None:
            self.parameters = Variable(torch.normal(mean = 0, std = 0.1, size = (self.n_qubits, 3)), requires_grad = True)

        for i in range(self.n_qubits):
            qml.RX(self.parameters[i, 0], wires=i)
            qml.RY(self.parameters[i, 1], wires=i)
            qml.RZ(self.parameters[i, 2], wires=i)

        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

class CircuitModel:
    def __init__(self, device: str, n_qubits: int):
        self.n_qubits = n_qubits
        self.device = device

    def build(self, n_layers:int):
        @qml.qnode(self.device, wires = self.n_qubits)
        parameters = Variable(torch.normal( mean=0. , std=0.1, size=(n_layers, self.n_qubits, 3)), requires_grad=True)
        for i in range(n_layers):
        




