import tensorflow as tf
import pennylane as qml


class QuantumCircuit:
    def __init__(self, num_qubits:int, dev: str):
        self.n_qubits = num_qubits
        self.dev = qml.device(dev, wires = self.n_qubits)

    def circuit_block(self, params, state=None):
        if state is not None:




