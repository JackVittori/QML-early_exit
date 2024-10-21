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
                    #qml.bit

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

        def _binary_quantum_function(params: Dict, state: torch.Tensor = None):
            mcasurements = []
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

            mcasurements.append(qml.measure(wires=0)) #measure first qubit

            for i in range(4, num_layers):
                for j in range(num_qubits):
                    qml.RX(params[f'layer_{i}'][j, 0], wires=j)
                    qml.RY(params[f'layer_{i}'][j, 1], wires=j)
                    qml.RZ(params[f'layer_{i}'][j, 2], wires=j)
                for j in range(num_qubits):
                    qml.CNOT(wires=[j, (j + 1) % num_qubits])

            mcasurements.append(qml.measure(wires=1)) #measure second qubit

            return mcasurements

        if ansatz == 'ansatz_1':
            self.quantum_node = _quantum_function

        elif ansatz == '2-class':
            self.quantum_node = _binary_quantum_function

    def forward(self, state: torch.Tensor = None):

        return self.quantum_node(params=self.params, state=state)

    def set_parameters(self, params: Dict):
        """Sets the params to a new value."""
        self.params = params

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

        elif ansatz == '2-class':
            @qml.qnode(device=self.dev, interface='torch')
            def _qnode_3(state: torch.Tensor):
                measurements = self.quantum_layer(state=state)
                mid_measurement, final_measurement = measurements
                return qml.probs(op=mid_measurement), qml.probs(op=final_measurement)
            self.quantum_node = _qnode_3
            self.ansatz = '2-class'

        else:
            raise ValueError("Please indicate an ansatz between 'ansatz_1', '2-class'")

    def ansatz(self):

        print(self.ansatz)
    def set_parameters(self, params: Dict):
        """Sets the params to a new value."""
        self.quantum_layer.set_parameters(params)
        self.params = params

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

    def draw(self, style: str = 'default', path: Optional[str] = None):
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
        fig, ax = qml.draw_mpl(self.quantum_node)(state=None)

        plt.show()
        if path is not None:
            fig.savefig(path)

    def fit(self, dataloader: DataLoader, sched_epochs: int, learning_rate:List[float], epochs: int,
            show_plot: Optional[bool] = False) -> tuple:

        fm_loss_function = torch.nn.NLLLoss()
        loss_history = list()
        fm_accuracy = list()
        optimizer = torch.optim.Adam(self.get_trainable_params(), lr=learning_rate[0])
        avg_time_per_epoch = 0

        for epoch in range(epochs):
            t_start = time()
            running_loss = 0
            mcm_accuracy_per_epoch = list()
            fm_accuracy_per_epoch = list()
            if epoch == sched_epochs:
                optimizer = torch.optim.Adam(self.get_trainable_params(), lr=learning_rate[1])

            with tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}/{epochs}') as tqdm_epoch:

                for _, (data, targets) in tqdm_epoch:
                    data = data / torch.linalg.norm(data, dim=1).view(-1, 1)

                    optimizer.zero_grad()

                    mcm_probs, final_probs = self.forward(state=data)

                    # fm computation
                    fm_predictions = torch.argmax(final_probs, dim=1)
                    fm_batch_accuracy = torch.sum(fm_predictions == targets).item() / len(fm_predictions)
                    fm_accuracy_per_epoch.append(fm_batch_accuracy)
                    fm_log_probs = torch.log(final_probs)
                    loss = fm_loss_function(fm_log_probs, targets)

                    running_loss += loss.item()

                    loss.backward()

                    optimizer.step()

                    tqdm_epoch.set_postfix(loss=loss.item(), fm_accuracy=fm_batch_accuracy)

            avg_time_per_epoch += time() - t_start

            loss_history.append(running_loss / len(dataloader))

            fm_accuracy.append(sum(fm_accuracy_per_epoch) / len(fm_accuracy_per_epoch))

            # print the time
            print("Time per epoch (s): ", time() - t_start)

            # print the loss
            print("Epoch: ", epoch + 1, "Loss: ", loss_history[epoch])

            print("--------------------------------------------------------------------------")
            print("Final Measurement accuracy: ", fm_accuracy[epoch])

            print("--------------------------------------------------------------------------")

        if show_plot:
            plt.style.use('ggplot')
            fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(20, 5))

            # Plotting the loss on the first subplot
            ax1.plot(list(range(epochs)), loss_history, marker='o', linestyle='-', color='b', label='Loss per Epoch')
            ax1.set_title('Training Loss Over Epochs', fontsize=16)
            ax1.set_xlabel('Epochs', fontsize=14)
            ax1.set_ylabel('Loss', fontsize=14)
            ax1.set_xticks(list(range(epochs)))
            ax1.legend()
            ax1.grid(True)


            ax3.plot(list(range(epochs)), fm_accuracy, marker='x', linestyle='--', color='r', label='Accuracy per Epoch')
            ax3.set_title('Final circuit Training Accuracy Over Epochs', fontsize=16)
            ax3.set_xlabel('Epochs', fontsize=14)
            ax3.set_ylabel('Accuracy', fontsize=14)
            ax3.set_xticks(list(range(epochs)))
            ax3.legend()
            ax3.grid(True)


            plt.tight_layout()
            plt.savefig('training_performance_plots.png', dpi=300)
            plt.show()

        return fm_accuracy, loss_history
