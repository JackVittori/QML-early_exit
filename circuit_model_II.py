import torch
import pennylane as qml
from torch.nn import Module
import matplotlib.pyplot as plt
from torch.autograd import Variable
import warnings
from typing import Optional, Dict, List


class QuantumCircuit(Module):
    """Quantum circuit using PennyLane and integrated with PyTorch."""

    def __init__(self, num_qubits: int, num_layers: int, interface: str = 'torch'):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.params = torch.nn.ParameterDict({
            f'layer_{i + 1}': torch.nn.Parameter(torch.rand(num_qubits, 3, requires_grad=True))
            for i in range(num_layers)
        })

        @qml.qnode(self.dev, interface=interface)
        def _quantum_function(params: Dict, state=None):
            if state is not None:
                qml.QubitStateVector(state, wires=range(self.num_qubits))

            for i in range(self.num_layers):
                for j in range(self.num_qubits):
                    qml.RX(params[f'layer_{i + 1}'][j, 0], wires=j)
                    qml.RY(params[f'layer_{i + 1}'][j, 1], wires=j)
                    qml.RZ(params[f'layer_{i + 1}'][j, 2], wires=j)
                for j in range(self.num_qubits):
                    qml.CNOT(wires=[j, (j + 1) % self.num_qubits])
            return qml.state()

        self.quantum_node = _quantum_function

    def forward(self, state=None):
        return self.quantum_node(self.params, state=state)


class FullQuantumModel(Module):
    """Full quantum model that integrates a quantum layer with measurement."""

    def __init__(self, qubits: int, layers: int):
        super().__init__()
        self.quantum_layer = QuantumCircuit(qubits, layers)
        self.params = self.quantum_layer.params

    def forward(self, state):
        """Calculate the probability distribution from quantum state measurements."""
        state_vector = self.quantum_layer(state=state)
        probabilities = torch.sum(torch.abs(state_vector[:, :2 ** (self.quantum_layer.num_qubits - 1)]) ** 2, dim=1)
        return probabilities.type(torch.float32)

    def trainable_layers(self):
        """Prints layers and their trainability status."""
        trainable_layers = dict()
        for layer in list(self.params.keys()):
            trainable_layers[layer] = self.params[layer].requires_grad
        print(trainable_layers)

    def trainable_parameters(self):
        """Prints number of trainable parameters."""
        print("Trainable parameters:", sum(p.numel() for p in self.parameters() if p.requires_grad == True))

    def get_trainable_params(self):
        """Retrieve parameters that are marked as trainable."""
        return iter([p for p in self.parameters() if p.requires_grad])

    def freeze_layers(self, layers: List[int]):
        """Freeze specified layers from training."""
        valid_keys = list(self.params.keys())

        for layer in layers:
            layer_name = f'layer_{layer}'
            if layer_name not in valid_keys:
                error_message = f"Please indicate the indexes of the layers choosing belong the following {valid_keys}"
                raise ValueError(error_message)
            if not self.params[layer_name].requires_grad:
                warnings.warn(f"{layer_name} has already been frozen (requires_grad = False).")
            else:
                self.params[layer_name].requires_grad = False

    def unfreeze_layers(self, layers: List[int]):
        """Unfreeze specified layers for training."""
        valid_keys = list(self.params.keys())

        for layer in layers:
            layer_name = f'layer_{layer}'
            if layer_name not in valid_keys:
                error_message = f"Please indicate the indexes of the layers choosing belong the following {valid_keys}"
                raise ValueError(error_message)
            if self.params[layer_name].requires_grad:
                warnings.warn(f"{layer_name} is already trainable (requires_grad = True).")
            else:
                self.params[layer_name].requires_grad = True

    def draw(self, style: str = 'default'):
        """Draws the quantum circuit with specified style."""
        valid_styles = {'black_white', 'black_white_dark', 'sketch', 'pennylane', 'pennylane_sketch',
                        'sketch_dark', 'solarized_light', 'solarized_dark', 'default'}

        # Check if the provided style is valid
        if style not in valid_styles:
            raise ValueError(f"Invalid style '{style}'. Valid styles are: {', '.join(valid_styles)}")
        qml.drawer.use_style(style)
        fig, ax = qml.draw_mpl(self.quantum_layer.quantum_node)(self.quantum_layer.params)
        plt.show()
