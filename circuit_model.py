import torch
import pennylane as qml
from torch.nn import Module
import matplotlib.pyplot as plt
from torch.autograd import Variable


class QuantumCircuit(Module):
    def __init__(self, num_qubits, num_layers, interface='torch'):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.params = torch.nn.Parameter(torch.rand(num_layers, num_qubits, 3, requires_grad=True))
        self.dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(self.dev, interface=interface)
        def _quantum_function(params, state=None):
            if state is not None:
                qml.QubitStateVector(state, wires=range(self.num_qubits))

            for i in range(self.num_layers):
                for j in range(self.num_qubits):
                    qml.RX(params[i, j, 0], wires=j)
                    qml.RY(params[i, j, 1], wires=j)
                    qml.RZ(params[i, j, 2], wires=j)
                for j in range(num_qubits):
                    qml.CNOT(wires=[j, (j + 1) % num_qubits])
            return qml.state()

        self.quantum_node = _quantum_function

    def forward(self, state=None):
        return self.quantum_node(self.params, state=state)


class FullQuantumModel(Module):
    def __init__(self, qubits, layers):
        super().__init__()
        self.quantum_layer = QuantumCircuit(qubits, layers)
        self.params = self.quantum_layer.params

    def forward(self, state):
        state_vector = self.quantum_layer(state)
        probabilities = torch.sum(torch.abs(state_vector[:, :2 ** (self.quantum_layer.num_qubits - 1)]) ** 2, dim=1)
        return probabilities.type(torch.float32)

    def draw(self, style='default'):
        valid_styles = {'black_white', 'black_white_dark', 'sketch', 'pennylane', 'pennylane_sketch',
                        'sketch_dark', 'solarized_light', 'solarized_dark', 'default'}

        # Check if the provided style is valid
        if style not in valid_styles:
            raise ValueError(f"Invalid style '{style}'. Valid styles are: {', '.join(valid_styles)}")
        qml.drawer.use_style(style)
        fig, ax = qml.draw_mpl(self.quantum_layer.quantum_node)(self.quantum_layer.params)
        plt.show()
