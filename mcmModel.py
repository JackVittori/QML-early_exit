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


class MCMCircuit(Module):
    """
    MCMCircuit class defining quantum computation integrating Pennylane with Pytorch and containing quantum circuit logic.
    """
    params: ParameterDict
    dev: Device
    num_qubits: int
    num_layers: int
    num_shots: int

    def __init__(self, num_qubits: int, num_layers: int, num_shots: int, ansatz: int = 'ansatz_1', interface: str = 'torch'):

        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_shots = num_shots
        # self.dev = qml.device("default.qubit", wires=num_qubits, shots=num_shots)
        self.params = torch.nn.ParameterDict({
            f'layer_{i}': torch.nn.Parameter(torch.rand(num_qubits, 3, requires_grad=True))
            for i in range(num_layers)
        })
        def _quantum_function(params: Dict, state: torch.Tensor = None):
            first_pair = [0, 1]
            second_pair = [2, 3]
            mcms = []

            if state is not None:
                # state vector initialization with input
                qml.QubitStateVector(state, wires=range(self.num_qubits))

            for i in range(4):
                for j in range(num_qubits):
                    qml.RX(params[f'layer_{i}'][j, 0], wires=j)
                    qml.RY(params[f'layer_{i}'][j, 1], wires=j)
                    qml.RZ(params[f'layer_{i}'][j, 2], wires=j)
                for j in range(num_qubits):
                    qml.CNOT(wires=[j, (j + 1) % num_qubits])

            for w in first_pair:
                mcms.append(qml.measure(wires=w))

            for i in range(4, num_layers):
                for j in range(num_qubits):
                    qml.RX(params[f'layer_{i}'][j, 0], wires=j)
                    qml.RY(params[f'layer_{i}'][j, 1], wires=j)
                    qml.RZ(params[f'layer_{i}'][j, 2], wires=j)
                for j in range(num_qubits):
                    qml.CNOT(wires=[j, (j + 1) % num_qubits])

            for w in second_pair:
                mcms.append(qml.measure(wires=w))

            return mcms

        def _quantum_function_2(params: Dict, state: torch.Tensor = None):
            first_pair = [0, 1]

            if state is not None:
                # state vector initialization with input
                qml.QubitStateVector(state, wires=range(self.num_qubits))

            for i in range(4):
                for j in range(num_qubits):
                    qml.RX(params[f'layer_{i}'][j, 0], wires=j)
                    qml.RY(params[f'layer_{i}'][j, 1], wires=j)
                    qml.RZ(params[f'layer_{i}'][j, 2], wires=j)
                for j in range(num_qubits):
                    qml.CNOT(wires=[j, (j + 1) % num_qubits])

            mcm = [qml.measure(wire) for wire in first_pair]
            for i in range(4, num_layers):
                for j in range(num_qubits):
                    qml.RX(params[f'layer_{i}'][j, 0], wires=j)
                    qml.RY(params[f'layer_{i}'][j, 1], wires=j)
                    qml.RZ(params[f'layer_{i}'][j, 2], wires=j)
                for j in range(num_qubits):
                    qml.CNOT(wires=[j, (j + 1) % num_qubits])

            return mcm

        if ansatz == 'ansatz_1':
            self.quantum_node = _quantum_function
        else:
            self.quantum_node = _quantum_function_2

    def forward(self, state: torch.Tensor = None):

        return self.quantum_node(params=self.params, state=state)


class MCMQuantumModel(Module):
    quantum_layer: MCMCircuit
    params: ParameterDict
    num_qubits: int
    num_layers: int
    num_shots: int
    dev: Device

    def __init__(self, qubits: int, layers: int, shots: Optional[int] = None, ansatz: int = 'ansatz_1'):

        super().__init__()
        self.quantum_layer = MCMCircuit(qubits, layers, shots, ansatz)
        self.params = self.quantum_layer.params
        self.num_qubits = self.quantum_layer.num_qubits
        self.num_layers = self.quantum_layer.num_layers
        self.num_shots = self.quantum_layer.num_shots
        if shots is None:
            self.dev = qml.device("default.qubit", wires=self.num_qubits)
        else:
            self.dev = qml.device("default.qubit", wires=self.num_qubits, shots=self.num_shots)

        if ansatz == 'ansatz_1':
            @qml.qnode(device=self.dev, interface='torch')
            def _qnode(state: torch.Tensor):
                results = self.quantum_layer(state=state)
                mcm = results[:2]
                fm = results[2:]
                return qml.probs(op=mcm), qml.probs(op=fm)

            self.quantum_node = _qnode
            self.ansatz = "Ansatz_1"

        elif ansatz == 'ansatz_2':
            @qml.qnode(device=self.dev, interface='torch')
            def _qnode_2(state: torch.Tensor):
                second_pair = [2, 3]
                mcm = self.quantum_layer(state=state)
                return qml.probs(op=mcm), qml.probs(wires=second_pair)

            self.quantum_node = _qnode_2
            self.ansatz = "Ansatz_2"

        else:
            raise ValueError("Please indicate an ansatz between 'ansatz_1' and 'ansatz_2'")

    def ansatz(self):

        print(self.ansatz)

    def forward(self, state: torch.Tensor):
        first_probs, second_probs = self.quantum_node(state=state)
        return first_probs, second_probs

    def trainable_layers(self):
        trainable_layers = dict()
        for layer in list(self.params.keys()):
            trainable_layers[layer] = self.params[layer].requires_grad
        print(trainable_layers)

    def trainable_parameters(self):

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
            show_plot: Optional[bool] = False) -> tuple:

        mcm_loss_function = torch.nn.NLLLoss()  # Negative Log Likelihood for multinomial classification
        fm_loss_function = torch.nn.NLLLoss()
        loss_history = list()
        mcm_accuracy = list()
        fm_accuracy = list()
        optimizer = torch.optim.Adam(self.get_trainable_params(), lr=learning_rate)
        avg_time_per_epoch = 0

        for epoch in range(epochs):
            t_start = time()
            running_loss = 0
            mcm_accuracy_per_epoch = list()
            fm_accuracy_per_epoch = list()

            with tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}/{epochs}') as tqdm_epoch:

                for _, (data, targets) in tqdm_epoch:
                    data = data / torch.linalg.norm(data, dim=1).view(-1, 1)

                    optimizer.zero_grad()

                    mcm_probs, final_probs = self.forward(state=data)

                    # mcm computation
                    mcm_predictions = torch.argmax(mcm_probs, dim=1)

                    mcm_batch_accuracy = torch.sum(mcm_predictions == targets).item() / len(mcm_predictions)

                    mcm_accuracy_per_epoch.append(mcm_batch_accuracy)
                    mcm_log_probs = torch.log(mcm_probs)
                    mcm_loss = mcm_loss_function(mcm_log_probs, targets)

                    # fm computation
                    fm_predictions = torch.argmax(final_probs, dim=1)
                    fm_batch_accuracy = torch.sum(fm_predictions == targets).item() / len(fm_predictions)
                    fm_accuracy_per_epoch.append(fm_batch_accuracy)
                    fm_log_probs = torch.log(final_probs)
                    fm_loss = fm_loss_function(fm_log_probs, targets)

                    # total loss
                    loss = mcm_loss + fm_loss

                    running_loss += loss.item()

                    loss.backward()

                    optimizer.step()

                    tqdm_epoch.set_postfix(loss=loss.item(),
                                           mcm_accuracy=mcm_batch_accuracy, fm_accuracy=fm_batch_accuracy)

            avg_time_per_epoch += time() - t_start

            loss_history.append(running_loss / len(dataloader))

            mcm_accuracy.append(sum(mcm_accuracy_per_epoch) / len(mcm_accuracy_per_epoch))
            fm_accuracy.append(sum(fm_accuracy_per_epoch) / len(fm_accuracy_per_epoch))

            # print the time
            print("Time per epoch: ", time() - t_start)

            # print the loss
            print("Epoch: ", epoch + 1, "Loss: ", loss_history[epoch])

            # print the accuracy
            print("Mid circuit accuracy: ", accuracy[epoch])

            print("--------------------------------------------------------------------------")
            print("Final Measurement accuracy: ", accuracy[epoch])

            print("--------------------------------------------------------------------------")

        if show_plot:
            plt.style.use('ggplot')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

            # Plotting the loss on the first subplot
            ax1.plot(list(range(epochs)), loss_history, marker='o', linestyle='-', color='b', label='Loss per Epoch')
            ax1.set_title('Training Loss Over Epochs', fontsize=16)
            ax1.set_xlabel('Epochs', fontsize=14)
            ax1.set_ylabel('Loss', fontsize=14)
            ax1.set_xticks(list(range(epochs)))
            ax1.legend()
            ax1.grid(True)

            # Plotting the accuracy on the second subplot
            ax2.plot(list(range(epochs)), accuracy, marker='x', linestyle='--', color='r', label='Accuracy per Epoch')
            ax2.set_title('Training Accuracy Over Epochs', fontsize=16)
            ax2.set_xlabel('Epochs', fontsize=14)
            ax2.set_ylabel('Accuracy', fontsize=14)
            ax2.set_xticks(list(range(epochs)))
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plt.savefig('training_performance_plots.png', dpi=300)
            plt.show()

        return mcm_accuracy, fm_accuracy, loss_history