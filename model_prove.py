import torch
import pennylane as qml
from pennylane.measurements import StateMP
from torch.nn import Module
import matplotlib.pyplot as plt
import warnings
from typing import Optional, Dict, List
from torch.utils.data import DataLoader
from time import time
from tqdm import tqdm
import math

class FullQuantumModel(Module):
    """
    Full quantum model class.
    """

    def __init__(self, num_qubits: int, num_layers: int, interface:str = 'torch', num_classes: int):
        """

        """
        super().__init__()
        self.required_qubits = math.ceil(math.log2(num_classes))
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.params = torch.nn.ParameterDict({
            f'layer_{i}': torch.nn.Parameter(torch.rand(num_qubits, 3, requires_grad=True))
            for i in range(num_layers)
        })

        @qml.qnode(self.dev, interface=interface)

    def forward(self, state: torch.Tensor = None, num_layers_to_execute: Optional[int] = None):
        if state is not None:
            qml.QubitStateVector(state, wires = range(self.num_qubits))

        # execute only a specified number of layers or all layers
        layers_to_execute = num_layers_to_execute if num_layers_to_execute is not None else self.num_layers

        for i in range(layers_to_execute):
            for j in range(self.num_qubits):
                qml.RX(self.params[f'layer_{i}'][j, 0], wires=j)
                qml.RY(self.params[f'layer_{i}'][j, 1], wires=j)
                qml.RZ(self.params[f'layer_{i}'][j, 2], wires=j)
            for j in range(self.num_qubits):
                qml.CNOT(wires=[j, (j + 1) % self.num_qubits])

        state_vector = qml.state()
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
            num_layers_to_execute: Optional[int] = None, show_plot: Optional[bool] = False) -> tuple:
        """
        Train the quantum circuit given a set of training data.

        :param show_plot: if true, show the plot of the loss and the accuracy.
        :param dataloader: dataloader with training data.
        :param learning_rate: learning rate of the optimizer.
        :param epochs: number of epochs.
        :param loss_function: loss function.
        :param num_layers_to_execute: The number of layers to execute
        :param show_plot: If True, plot the loss during training.
        :return: tuple containing accuracy history and loss history per epoch.
        """

        if num_layers_to_execute is not None:
            self.unfreeze_layers(list(range(self.num_layers)))
            self.freeze_layers(
                [element for element in range(self.num_layers) if element not in range(num_layers_to_execute)])

        loss_history = list()
        accuracy = list()
        optimizer = torch.optim.Adam(self.get_trainable_params(), lr=learning_rate)
        avg_time_per_epoch = 0

        for epoch in range(epochs):
            t_start = time()
            running_loss = 0
            predictions_per_epoch = list()
            targets_per_epoch = list()

            with tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}/{epochs}') as tqdm_epoch:

                for _, (data, targets) in tqdm_epoch:
                    data = data / torch.linalg.norm(data, dim=1).view(-1, 1)

                    optimizer.zero_grad()

                    output = self.forward(state=data, num_layers_to_execute=num_layers_to_execute)

                    predictions_per_epoch.append((output > 0.5))

                    targets_per_epoch.append(targets)

                    loss = loss_function(output, targets)

                    running_loss += loss.item()

                    loss.backward()

                    optimizer.step()

                    tqdm_epoch.set_postfix(loss=loss.item(),
                                           accuracy=torch.sum((output > 0.5) == targets).item() / len(output))

            avg_time_per_epoch += time() - t_start

            loss_history.append(running_loss / len(dataloader))

            predictions_per_epoch = torch.cat(predictions_per_epoch, dim=0)

            targets_per_epoch = torch.cat(targets_per_epoch, dim=0)

            accuracy.append(
                torch.sum(predictions_per_epoch == targets_per_epoch).item() / len(predictions_per_epoch))

            # print the time
            print("Time per epoch: ", time() - t_start)

            # print the loss
            print("Epoch: ", epoch, "Loss: ", loss_history[epoch])

            # print the accuracy
            print("Accuracy: ", accuracy[epoch])

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

        return accuracy, loss_history

    #def test(self, dataloader: DataLoader, loss_function: torch.nn.modules.loss = torch.nn.BCELoss(),
    #num_layers_to_execute: Optional[int] = None, show_plot: Optional[bool] = False):

    #self.freeze_layers(list(range(self.num_layers)))

    #self.eval()

    #for _, (data, targets) in dataloader:

    #data = data / torch.linalg.norm(data, dim=1).view(-1, 1)

    #output = self.forward(state=data, num_layers_to_execute=num_layers_to_execute)