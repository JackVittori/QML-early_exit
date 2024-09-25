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
    early_exits: List[int]  #early-exit indexes
    list_of_pairs: List[List[int]]  #pair of qubits to measure

    def __init__(self, num_qubits: int, num_layers: int, early_exits: List[int]):
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
                raise ValueError(f"Error: The early exit {num} exceeds the number of layers ({self.num_layers})")

        if len(self.early_exits) > 3:
            raise ValueError(f"Error: More than 3 early exits are not supported due to max 8 qubits limitation")

        self.list_of_pairs = []
        start_value = 0
        for i in range(len(self.early_exits)):
            pair = [start_value, start_value + 1]
            self.list_of_pairs.append(pair)
            start_value += 2
        final_pair = [start_value, start_value + 1]
        self.list_of_pairs.append(final_pair)

        def _quantum_function(params: Dict, state: torch.Tensor = None):
            mcms = []
            measured_so_far = 0

            if state is not None:
                qml.QubitStateVector(state, wires=range(self.num_qubits))

            for i in range(self.num_layers):
                for j in range(self.num_qubits):
                    qml.RX(params[f'layer_{i}'][j, 0], wires=j)
                    qml.RY(params[f'layer_{i}'][j, 1], wires=j)
                    qml.RZ(params[f'layer_{i}'][j, 2], wires=j)

                for j in range(self.num_qubits):
                    qml.CNOT(wires=[j, (j + 1) % self.num_qubits])

                if i in self.early_exits:
                    current_pair = self.list_of_pairs[measured_so_far]
                    for w in current_pair:
                        mcms.append(qml.measure(wires=w))
                    measured_so_far += 1

            final_pair = self.list_of_pairs[-1]
            for w in final_pair:
                mcms.append(qml.measure(wires=w))

            return mcms

        self.quantum_node = _quantum_function

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
    num_exits: int
    dev: Device

    def __init__(self, qubits: int, layers: int, early_exits=List[int]):
        super().__init__()
        self.quantum_layer = MCMCircuit(num_qubits=qubits, num_layers=layers, early_exits=early_exits)
        self.params = self.quantum_layer.params
        self.num_qubits = self.quantum_layer.num_qubits
        self.num_layers = self.quantum_layer.num_layers
        self.early_exits = early_exits
        self.dev = qml.device("default.qubit", wires=self.num_qubits)
        self.num_exits = len(self.early_exits)

        @qml.qnode(device=self.dev, interface='torch')
        def _qnode(state: torch.Tensor):
            results = self.quantum_layer(state=state)
            expected_length = 2 * (self.num_exits + 1)
            assert len(results) == expected_length, (f"Number of measurements ({len(results)}) do not correspond to Early Exits ({expected_length}).")
            blocks = [results[2 * i: 2 * (i + 1)] for i in range(self.num_exits)]
            fm = results[-2:]
            return [qml.probs(op=block) for block in blocks] + [qml.probs(op=fm)]

        self.quantum_node = _qnode

    def set_parameters(self, params: Dict):
        """Sets the params to a new value."""
        self.quantum_layer.set_parameters(params)
        self.params = params

    def forward(self, state: torch.Tensor):
        probs = self.quantum_node(state=state)

        if self.num_exits == 1:
            return probs[0], probs[1]
        elif self.num_exits == 2:
            return probs[0], probs[1], probs[2]
        else:
            return probs[0], probs[1], probs[2], probs[3]

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


    def fit(self, dataloader: DataLoader, learning_rate: float, epochs: int, show_plot: Optional[bool] = False) -> tuple:

        if self.num_exits == 1:
            mcm_loss_function = torch.nn.NLLLoss()
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

                with tqdm(enumerate(dataloader), total=len(dataloader),
                          desc=f'Epoch {epoch + 1}/{epochs}') as tqdm_epoch:

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
                print("Time per epoch (s): ", time() - t_start)

                # print the loss
                print("Epoch: ", epoch + 1, "Loss: ", loss_history[epoch])

                print("--------------------------------------------------------------------------")
                # print the accuracy
                print("Mid circuit accuracy: ", mcm_accuracy[epoch])

                print("--------------------------------------------------------------------------")
                print("Final Measurement accuracy: ", fm_accuracy[epoch])

                print("--------------------------------------------------------------------------")

            if show_plot:
                plt.style.use('ggplot')
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

                # Plotting the loss on the first subplot
                ax1.plot(list(range(epochs)), loss_history, marker='o', linestyle='-', color='b',
                         label='Loss per Epoch')
                ax1.set_title('Training Loss Over Epochs', fontsize=16)
                ax1.set_xlabel('Epochs', fontsize=14)
                ax1.set_ylabel('Loss', fontsize=14)
                ax1.set_xticks(list(range(epochs)))
                ax1.legend()
                ax1.grid(True)

                # Plotting the accuracy on the second subplot
                ax2.plot(list(range(epochs)), mcm_accuracy, marker='x', linestyle='--', color='r',
                         label='Accuracy per Epoch')
                ax2.set_title('Early Exit Training Accuracy Over Epochs', fontsize=16)
                ax2.set_xlabel('Epochs', fontsize=14)
                ax2.set_ylabel('Accuracy', fontsize=14)
                ax2.set_xticks(list(range(epochs)))
                ax2.legend()
                ax2.grid(True)

                ax3.plot(list(range(epochs)), fm_accuracy, marker='x', linestyle='--', color='r',
                         label='Accuracy per Epoch')
                ax3.set_title('Final circuit Training Accuracy Over Epochs', fontsize=16)
                ax3.set_xlabel('Epochs', fontsize=14)
                ax3.set_ylabel('Accuracy', fontsize=14)
                ax3.set_xticks(list(range(epochs)))
                ax3.legend()
                ax3.grid(True)

                plt.tight_layout()
                plt.savefig('training_performance_plots.png', dpi=300)
                plt.show()

            return mcm_accuracy, fm_accuracy, loss_history


        if self.num_exits == 2:
            early_loss_1 = torch.nn.NLLLoss()
            early_loss_2 = torch.nn.NLLLoss()
            fm_loss_function = torch.nn.NLLLoss()
            loss_history = list()
            early_1_accuracy = list()
            early_2_accuracy = list()
            fm_accuracy = list()
            optimizer = torch.optim.Adam(self.get_trainable_params(), lr=learning_rate)
            avg_time_per_epoch = 0

            for epoch in range(epochs):
                t_start = time()
                running_loss = 0
                early_1_accuracy_per_epoch = list()
                early_2_accuracy_per_epoch = list()
                fm_accuracy_per_epoch = list()

                with tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}/{epochs}') as tqdm_epoch:

                    for _, (data, targets) in tqdm_epoch:
                        data = data / torch.linalg.norm(data, dim=1).view(-1, 1)
                        optimizer.zero_grad()
                        early_1_probs, early_2_probs, final_probs = self.forward(state=data)

                        #early exit 1 computation
                        early_1_prediction = torch.argmax(early_1_probs, dim=1)
                        early_1_batch_accuracy = torch.sum(early_1_prediction == targets).item() / len(early_1_prediction)
                        early_1_accuracy_per_epoch.append(early_1_batch_accuracy)
                        early_1_log_probs = torch.log(early_1_probs)
                        early_1_loss = early_loss_1(early_1_log_probs, targets)

                        #early exit 2 computation

                        early_2_prediction = torch.argmax(early_2_probs, dim=1)
                        early_2_batch_accuracy = torch.sum(early_2_prediction == targets).item() / len(early_2_prediction)
                        early_2_accuracy_per_epoch.append(early_2_batch_accuracy)
                        early_2_log_probs = torch.log(early_2_probs)
                        early_2_loss = early_loss_2(early_2_log_probs, targets)

                        # fm computation
                        fm_predictions = torch.argmax(final_probs, dim=1)
                        fm_batch_accuracy = torch.sum(fm_predictions == targets).item() / len(fm_predictions)
                        fm_accuracy_per_epoch.append(fm_batch_accuracy)
                        fm_log_probs = torch.log(final_probs)
                        fm_loss = fm_loss_function(fm_log_probs, targets)

                        #loss computation
                        loss = early_1_loss + early_2_loss + fm_loss
                        running_loss += loss.item()

                        loss.backward()

                        optimizer.step()

                        tqdm_epoch.set_postfix(loss=loss.item(),
                                               early_exit_1_accuracy=early_1_batch_accuracy,
                                               early_exit_2_accuracy=early_2_batch_accuracy,
                                               fm_accuracy=fm_batch_accuracy)

                avg_time_per_epoch += time() - t_start

                loss_history.append(running_loss / len(dataloader))

                early_1_accuracy.append(sum(early_1_accuracy_per_epoch) / len(early_1_accuracy_per_epoch))
                early_2_accuracy.append(sum(early_2_accuracy_per_epoch) / len(early_2_accuracy_per_epoch))
                fm_accuracy.append(sum(fm_accuracy_per_epoch) / len(fm_accuracy_per_epoch))

                # print the time
                print("Time per epoch (s): ", time() - t_start)

                # print the loss
                print("Epoch: ", epoch + 1, "Loss: ", loss_history[epoch])

                print("--------------------------------------------------------------------------")
                # print the accuracy
                print("Earlu exit 1 accuracy: ", early_1_accuracy[epoch])

                print("--------------------------------------------------------------------------")
                print("Earlu exit 2 accuracy: ", early_2_accuracy[epoch])

                print("--------------------------------------------------------------------------")
                print("Final Measurement accuracy: ", fm_accuracy[epoch])

                print("--------------------------------------------------------------------------")

            if show_plot:
                plt.style.use('ggplot')
                fig, axs = plt.subplots(2, 3, figsize=(20, 10))  # 2 righe, 3 colonne

                # Plotting the loss on the first row, first subplot
                axs[0, 0].plot(list(range(epochs)), loss_history, marker='o', linestyle='-', color='b',
                               label='Loss per Epoch')
                axs[0, 0].set_title('Training Loss Over Epochs', fontsize=16)
                axs[0, 0].set_xlabel('Epochs', fontsize=14)
                axs[0, 0].set_ylabel('Loss', fontsize=14)
                axs[0, 0].set_xticks(list(range(epochs)))
                axs[0, 0].legend()
                axs[0, 0].grid(True)

                # Empty subplots on the first row (as requested only 1 plot for loss on first row)
                axs[0, 1].axis('off')  # Turn off subplot for aesthetics
                axs[0, 2].axis('off')  # Turn off subplot for aesthetics

                # Plotting early_accuracy_1 on the second row, first subplot
                axs[1, 0].plot(list(range(epochs)), early_1_accuracy, marker='x', linestyle='--', color='r',
                               label='Early Accuracy 1')
                axs[1, 0].set_title('Early Exit 1 Accuracy Over Epochs', fontsize=16)
                axs[1, 0].set_xlabel('Epochs', fontsize=14)
                axs[1, 0].set_ylabel('Accuracy', fontsize=14)
                axs[1, 0].set_xticks(list(range(epochs)))
                axs[1, 0].legend()
                axs[1, 0].grid(True)

                # Plotting early_accuracy_2 on the second row, second subplot
                axs[1, 1].plot(list(range(epochs)), early_2_accuracy, marker='x', linestyle='--', color='g',
                               label='Early Accuracy 2')
                axs[1, 1].set_title('Early Exit 2 Accuracy Over Epochs', fontsize=16)
                axs[1, 1].set_xlabel('Epochs', fontsize=14)
                axs[1, 1].set_ylabel('Accuracy', fontsize=14)
                axs[1, 1].set_xticks(list(range(epochs)))
                axs[1, 1].legend()
                axs[1, 1].grid(True)

                # Plotting fm_accuracy on the second row, third subplot
                axs[1, 2].plot(list(range(epochs)), fm_accuracy, marker='x', linestyle='--', color='b',
                               label='Final Model Accuracy')
                axs[1, 2].set_title('Final Model Accuracy Over Epochs', fontsize=16)
                axs[1, 2].set_xlabel('Epochs', fontsize=14)
                axs[1, 2].set_ylabel('Accuracy', fontsize=14)
                axs[1, 2].set_xticks(list(range(epochs)))
                axs[1, 2].legend()
                axs[1, 2].grid(True)

                plt.tight_layout()
                plt.savefig('training_performance_plots.png', dpi=300)
                plt.show()

            return early_1_accuracy, early_2_accuracy, fm_accuracy, loss_history

        if self.num_exits == 3:
            early_loss_1 = torch.nn.NLLLoss()
            early_loss_2 = torch.nn.NLLLoss()
            early_loss_3 = torch.nn.NLLLoss()
            fm_loss_function = torch.nn.NLLLoss()
            loss_history = list()
            early_1_accuracy = list()
            early_2_accuracy = list()
            early_3_accuracy = list()
            fm_accuracy = list()
            optimizer = torch.optim.Adam(self.get_trainable_params(), lr=learning_rate)
            avg_time_per_epoch = 0

            for epoch in range(epochs):
                t_start = time()
                running_loss = 0
                early_1_accuracy_per_epoch = list()
                early_2_accuracy_per_epoch = list()
                early_3_accuracy_per_epoch = list()
                fm_accuracy_per_epoch = list()

                with tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}/{epochs}') as tqdm_epoch:

                    for _, (data, targets) in tqdm_epoch:
                        data = data / torch.linalg.norm(data, dim=1).view(-1, 1)
                        optimizer.zero_grad()
                        early_1_probs, early_2_probs, early_3_probs, final_probs = self.forward(state=data)

                        #early exit 1 computation
                        early_1_prediction = torch.argmax(early_1_probs, dim=1)
                        early_1_batch_accuracy = torch.sum(early_1_prediction == targets).item() / len(early_1_prediction)
                        early_1_accuracy_per_epoch.append(early_1_batch_accuracy)
                        early_1_log_probs = torch.log(early_1_probs)
                        early_1_loss = early_loss_1(early_1_log_probs, targets)

                        #early exit 2 computation

                        early_2_prediction = torch.argmax(early_2_probs, dim=1)
                        early_2_batch_accuracy = torch.sum(early_2_prediction == targets).item() / len(early_2_prediction)
                        early_2_accuracy_per_epoch.append(early_2_batch_accuracy)
                        early_2_log_probs = torch.log(early_2_probs)
                        early_2_loss = early_loss_2(early_2_log_probs, targets)

                        # early exit 3 computation

                        early_3_prediction = torch.argmax(early_3_probs, dim=1)
                        early_3_batch_accuracy = torch.sum(early_3_prediction == targets).item() / len(early_3_prediction)
                        early_3_accuracy_per_epoch.append(early_3_batch_accuracy)
                        early_3_log_probs = torch.log(early_3_probs)
                        early_3_loss = early_loss_3(early_3_log_probs, targets)

                        # fm computation
                        fm_predictions = torch.argmax(final_probs, dim=1)
                        fm_batch_accuracy = torch.sum(fm_predictions == targets).item() / len(fm_predictions)
                        fm_accuracy_per_epoch.append(fm_batch_accuracy)
                        fm_log_probs = torch.log(final_probs)
                        fm_loss = fm_loss_function(fm_log_probs, targets)

                        #loss computation
                        loss = early_1_loss + early_2_loss + early_3_loss + fm_loss
                        running_loss += loss.item()

                        loss.backward()

                        optimizer.step()

                        tqdm_epoch.set_postfix(loss=loss.item(),
                                               early_exit_1_accuracy=early_1_batch_accuracy,
                                               early_exit_2_accuracy=early_2_batch_accuracy,
                                               early_exit_3_accuracy=early_3_batch_accuracy,
                                               fm_accuracy=fm_batch_accuracy)

                avg_time_per_epoch += time() - t_start

                loss_history.append(running_loss / len(dataloader))

                early_1_accuracy.append(sum(early_1_accuracy_per_epoch) / len(early_1_accuracy_per_epoch))
                early_2_accuracy.append(sum(early_2_accuracy_per_epoch) / len(early_2_accuracy_per_epoch))
                early_3_accuracy.append(sum(early_3_accuracy_per_epoch) / len(early_3_accuracy_per_epoch))
                fm_accuracy.append(sum(fm_accuracy_per_epoch) / len(fm_accuracy_per_epoch))

                # print the time
                print("Time per epoch (s): ", time() - t_start)

                # print the loss
                print("Epoch: ", epoch + 1, "Loss: ", loss_history[epoch])

                print("--------------------------------------------------------------------------")
                # print the accuracy
                print("Earlu exit 1 accuracy: ", early_1_accuracy[epoch])

                print("--------------------------------------------------------------------------")
                print("Earlu exit 2 accuracy: ", early_2_accuracy[epoch])

                print("--------------------------------------------------------------------------")

                print("Earlu exit 3 accuracy: ", early_3_accuracy[epoch])

                print("--------------------------------------------------------------------------")
                print("Final Measurement accuracy: ", fm_accuracy[epoch])

                print("--------------------------------------------------------------------------")

            if show_plot:
                plt.style.use('ggplot')
                fig, axs = plt.subplots(2, 3, figsize=(20, 10))  # 2 righe, 3 colonne

                # Plotting the loss on the first row, first subplot
                axs[0, 0].plot(list(range(epochs)), loss_history, marker='o', linestyle='-', color='b',
                               label='Loss per Epoch')
                axs[0, 0].set_title('Training Loss Over Epochs', fontsize=16)
                axs[0, 0].set_xlabel('Epochs', fontsize=14)
                axs[0, 0].set_ylabel('Loss', fontsize=14)
                axs[0, 0].set_xticks(list(range(epochs)))
                axs[0, 0].legend()
                axs[0, 0].grid(True)

                # Plotting early_accuracy_1 on the first row, second subplot
                axs[0, 1].plot(list(range(epochs)), early_1_accuracy, marker='x', linestyle='--', color='r',
                               label='Early Accuracy 1')
                axs[0, 1].set_title('Early Exit 1 Accuracy Over Epochs', fontsize=16)
                axs[0, 1].set_xlabel('Epochs', fontsize=14)
                axs[0, 1].set_ylabel('Accuracy', fontsize=14)
                axs[0, 1].set_xticks(list(range(epochs)))
                axs[0, 1].legend()
                axs[0, 1].grid(True)

                # Empty third subplot on the first row
                axs[0, 2].axis('off')  # Turn off subplot for aesthetics

                # Plotting early_accuracy_2 on the second row, first subplot
                axs[1, 0].plot(list(range(epochs)), early_2_accuracy, marker='x', linestyle='--', color='g',
                               label='Early Accuracy 2')
                axs[1, 0].set_title('Early Exit 2 Accuracy Over Epochs', fontsize=16)
                axs[1, 0].set_xlabel('Epochs', fontsize=14)
                axs[1, 0].set_ylabel('Accuracy', fontsize=14)
                axs[1, 0].set_xticks(list(range(epochs)))
                axs[1, 0].legend()
                axs[1, 0].grid(True)

                # Plotting early_accuracy_3 on the second row, second subplot
                axs[1, 1].plot(list(range(epochs)), early_3_accuracy, marker='x', linestyle='--', color='orange',
                               label='Early Accuracy 3')
                axs[1, 1].set_title('Early Exit 3 Accuracy Over Epochs', fontsize=16)
                axs[1, 1].set_xlabel('Epochs', fontsize=14)
                axs[1, 1].set_ylabel('Accuracy', fontsize=14)
                axs[1, 1].set_xticks(list(range(epochs)))
                axs[1, 1].legend()
                axs[1, 1].grid(True)

                # Plotting fm_accuracy on the second row, third subplot
                axs[1, 2].plot(list(range(epochs)), fm_accuracy, marker='x', linestyle='--', color='b',
                               label='Final Model Accuracy')
                axs[1, 2].set_title('Final Model Accuracy Over Epochs', fontsize=16)
                axs[1, 2].set_xlabel('Epochs', fontsize=14)
                axs[1, 2].set_ylabel('Accuracy', fontsize=14)
                axs[1, 2].set_xticks(list(range(epochs)))
                axs[1, 2].legend()
                axs[1, 2].grid(True)

                plt.tight_layout()
                plt.savefig('training_performance_plots.png', dpi=300)
                plt.show()

            return early_1_accuracy, early_2_accuracy, early_3_accuracy, fm_accuracy, loss_history