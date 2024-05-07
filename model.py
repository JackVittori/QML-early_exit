import torch
import pennylane as qml
from pennylane.measurements import StateMP
from torch.nn import Module
import matplotlib.pyplot as plt
from torch.autograd import Variable
import warnings
from typing import Optional, Dict, List
from torch.utils.data import DataLoader
from time import time
from tqdm import tqdm


class QuantumCircuit(Module):
    """Quantum circuit using PennyLane and integrated with PyTorch."""

    def __init__(self, num_qubits: int, num_layers: int, interface: str = 'torch'):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.params = torch.nn.ParameterDict({
            f'layer_{i}': torch.nn.Parameter(torch.rand(num_qubits, 3, requires_grad=True))
            for i in range(num_layers)
        })

        @qml.qnode(self.dev, interface=interface)
        def _quantum_function(params: Dict, state: torch.Tensor = None,
                              num_layers_to_execute: Optional[int] = None) -> StateMP:
            """
            Execute the quantum circuit with specified parameters and initial state up to the given number of layers.

            :param num_layers_to_execute: The number of layers to execute.
            :param params: Dictionary of parameters to be passed to the rotational gates.
            :param state: a tensor of shape (batch, 256) to initialize the circuit with the image (16x16).
            :return: the output state of the quantum circuit in the computational basis

            Each element of the state vector representing the quantum system is a complex number representing the
            probability amplitude of the system being in one of the 256 = 2^8 states. Thus, data has to be normalized
            before being passed to the quantum circuit.
            """
            if state is not None:
                # state vector initialization with input
                qml.QubitStateVector(state, wires=range(self.num_qubits))

            # execute only a specified number of layers or all layers
            layers_to_execute = num_layers_to_execute if num_layers_to_execute is not None else self.num_layers

            for i in range(layers_to_execute):
                for j in range(self.num_qubits):
                    qml.RX(params[f'layer_{i}'][j, 0], wires=j)
                    qml.RY(params[f'layer_{i}'][j, 1], wires=j)
                    qml.RZ(params[f'layer_{i}'][j, 2], wires=j)
                for j in range(self.num_qubits):
                    qml.CNOT(wires=[j, (j + 1) % self.num_qubits])
            return qml.state()

        self.quantum_node = _quantum_function

    def forward(self, state: torch.Tensor = None, num_layers_to_execute: Optional[int] = None) -> StateMP:
        """
        Perform a forward pass on the quantum circuit.

        :param state: a tensor of shape (batch, 256) to initialize the circuit with the image (16x16).
        :param num_layers_to_execute: The number of layers to execute.
        :return: the output state of the quantum circuit in the computational basis
        """
        return self.quantum_node(params=self.params, state=state, num_layers_to_execute=num_layers_to_execute)


class FullQuantumModel(Module):
    """
    Full quantum model class.
    """

    def __init__(self, qubits: int, layers: int):
        super().__init__()
        self.quantum_layer = QuantumCircuit(qubits, layers)
        self.params = self.quantum_layer.params
        self.num_qubits = self.quantum_layer.num_qubits
        self.num_layers = self.quantum_layer.num_layers

    def forward(self, state: torch.Tensor, num_layers_to_execute: Optional[int] = None):
        """
        Calculate the probability distribution from quantum state measurements.

        :param num_layers_to_execute: The number of layers to execute.
        :param state: a tensor of shape (batch, 256) to initialize the circuit with the image (16x16).
        :return: probability of being 1.
        """
        state_vector = self.quantum_layer(state=state, num_layers_to_execute=num_layers_to_execute)
        probabilities = torch.sum(torch.abs(state_vector[:, :2 ** (self.quantum_layer.num_qubits - 1)]) ** 2, dim=1)
        return probabilities.type(torch.float32)

    def trainable_layers(self):
        """
        Prints layers and their trainability status.
        """
        trainable_layers = dict()
        for layer in list(self.params.keys()):
            trainable_layers[layer] = self.params[layer].requires_grad
        print(trainable_layers)

    def trainable_parameters(self):
        """
        Prints number of trainable parameters.
        """
        print("Trainable parameters:", sum(p.numel() for p in self.parameters() if p.requires_grad == True))

    def get_trainable_params(self):
        """
        Retrieve parameters that are marked as trainable.

        :return: iterable of parameter marked as trainable.
        """
        return iter([p for p in self.parameters() if p.requires_grad])

    def freeze_layers(self, layers: List[int]):
        """
        Freeze specified layers from training.

        :param layers: List of indexes of layers to freeze.
        """
        valid_keys = list(self.params.keys())

        for layer in layers:
            layer_name = f'layer_{layer}'
            if layer_name not in valid_keys:
                error_message = f"Please indicate the indexes of the layers choosing belong the following {valid_keys}"
                raise ValueError(error_message)
            else:
                self.params[layer_name].requires_grad = False

    def unfreeze_layers(self, layers: List[int]):
        """
        Unfreeze specified layers for training.

        :param layers: List of indexes of layers that has to be set to non trainable.

        """
        valid_keys = list(self.params.keys())

        for layer in layers:
            layer_name = f'layer_{layer}'
            if layer_name not in valid_keys:
                error_message = f"Please indicate the indexes of the layers choosing belong the following {valid_keys}"
                raise ValueError(error_message)
            else:
                self.params[layer_name].requires_grad = True

    def draw(self, style: str = 'default'):
        """
        Draw the quantum circuit with specified style.

        :param style: style of drawing circuit.
        """

        valid_styles = {'black_white', 'black_white_dark', 'sketch', 'pennylane', 'pennylane_sketch',
                        'sketch_dark', 'solarized_light', 'solarized_dark', 'default'}

        # Check if the provided style is valid
        if style not in valid_styles:
            raise ValueError(f"Invalid style '{style}'. Valid styles are: {', '.join(valid_styles)}")
        qml.drawer.use_style(style)
        fig, ax = qml.draw_mpl(self.quantum_layer.quantum_node)(self.quantum_layer.params)
        plt.show()

    def fit(self, dataloader: DataLoader, learning_rate: float, epochs: int,
            loss_function: torch.nn.modules.loss = torch.nn.BCELoss(),
            num_layers_to_execute: Optional[int] = None) -> tuple:
        """
        Train the quantum circuit given a set of training data.

        :param dataloader: dataloader with training data.
        :param learning_rate: learning rate of the optimizer.
        :param epochs: number of epochs.
        :param loss_function: loss function.
        :param num_layers_to_execute: The number of layers to execute
        :return: tuple containing average time per epoch and loss history.
        """

        if num_layers_to_execute is not None:
            self.unfreeze_layers(list(range(self.num_layers)))
            self.freeze_layers([element for element in range(self.num_layers) if element not in range(num_layers_to_execute)])

        loss_history = list()
        optimizer = torch.optim.Adam(self.get_trainable_params(), lr=learning_rate)
        avg_time_per_epoch = 0

        for epoch in range(epochs):
            t_start = time()

            with tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}/{epochs}') as tqdm_epoch:

                for _, (data, targets) in tqdm_epoch:
                    data = data / torch.linalg.norm(data, dim=1).view(-1, 1)

                    optimizer.zero_grad()

                    output = self.forward(state=data, num_layers_to_execute=num_layers_to_execute)

                    loss = loss_function(output, targets)

                    loss.backward()

                    optimizer.step()

                    tqdm_epoch.set_postfix(loss=loss.item(),
                                           accuracy=torch.sum((output > 0.5) == targets).item() / dataloader.batch_size)

            avg_time_per_epoch += time() - t_start

            loss_history.append(loss.item())

            # print the time
            print("Time per epoch: ", time() - t_start)

            # print the loss
            print("Epoch: ", epoch, "Loss: ", loss.item())

            # print the accuracy
            print("Accuracy: ", torch.sum((output > 0.5) == targets).item() / dataloader.batch_size)

            print("--------------------------------------------------------------------------")

        return avg_time_per_epoch / epochs, loss_history

