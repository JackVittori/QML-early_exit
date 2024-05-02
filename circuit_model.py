import torch
import pennylane as qml
from torch.nn import Module

NUM_QUBITS = 8
NUM_LAYERS = 3


class QuantumCircuit(Module):
    def __init__(self, num_qubits, num_layers, interface='torch'):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.params = torch.nn.Parameter(torch.rand(num_layers, num_qubits, 3, requires_grad=True))
        self.dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(self.dev, interface=interface)
        def qfunc(params, state=None):
            if state is not None:
                qml.QubitStateVector(state, wires=range(num_qubits))

            for i in range(num_layers):
                for j in range(num_qubits):
                    qml.RX(params[i, j, 0], wires=j)
                    qml.RY(params[i, j, 1], wires=j)
                    qml.RZ(params[i, j, 2], wires=j)
                for j in range(num_qubits):
                    qml.CNOT(wires=[j, (j + 1) % num_qubits])
            return qml.state()

        self.qfunc = qfunc

    def forward(self, state=None):
        return self.qfunc(self.params, state=state)


class FullQuantumModel(Module):
    def __init__(self):
        super().__init__()
        self.quantum_layer = QuantumCircuit(NUM_QUBITS, NUM_LAYERS)

    def forward(self, state):
        state_vector = self.quantum_layer(state)
        probabilities = torch.sum(torch.abs(state_vector[:, :2 ** (NUM_QUBITS - 1)]) ** 2, dim=1)
        return probabilities.type(torch.float32)


# Instantiate model
model = FullQuantumModel()

# Example of disabling gradients for specific layers
for i in range(NUM_LAYERS):
    if i % 2 == 0:  # Disable gradient for even layers
        model.quantum_layer.params.data[i].requires_grad_(False)

# Example usage
initial_state = torch.rand(2 ** NUM_QUBITS)  # Random initial state
initial_state /= torch.norm(initial_state)  # Normalize

output_probabilities = model(initial_state)
print(output_probabilities)
