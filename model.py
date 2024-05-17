import torch
import pennylane as qml
from pennylane import Device
from pennylane.measurements import StateMP
from torch.nn import Module, ParameterDict
import matplotlib.pyplot as plt
import warnings
from typing import Optional, Dict, List, Any
from torch.utils.data import DataLoader
from time import time
from tqdm import tqdm
import math


class QuantumCircuit(Module):
    """
    QuantumCircuit class defining quantum computation integrating Pennylane with Pytorch and containing quantum
    circuit logic.
    """
    params: ParameterDict
    dev: Device
    num_layers: int
    num_qubits: int

    def __init__(self, num_qubits: int, num_layers: int, interface: str = 'torch'):
        """
        Initialize the QuantumCircuit class.

        :param num_qubits: Number of qubits in the quantum circuit.
        :param num_layers: Number of layers in the quantum circuit.
        :param interface: Interface to use with the quantum node. Defaults to 'torch'.
        """
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
    FullQuantumModel builds upon QuantumCircuit class to create a trainable model and containing machine learning model
    logic.
    """
    quantum_layer: QuantumCircuit
    params: ParameterDict
    num_qubits: int
    num_layers: int
    num_classes: int
    classification_qubits: int

    def __init__(self, qubits: int, layers: int, num_classes: int):
        """
        Initialize the FullQuantumModel class.

        :param qubits: Number of qubits in the quantum circuit.
        :param layers: Number of layers in the quantum circuit.
        :param num_classes: Number of classes in the dataset.
        """
        super().__init__()
        self.quantum_layer = QuantumCircuit(qubits, layers)
        self.params = self.quantum_layer.params
        self.num_qubits = self.quantum_layer.num_qubits
        self.num_layers = self.quantum_layer.num_layers
        self.num_classes = num_classes
        self.classification_qubits = math.ceil(math.log2(num_classes))
        if self.classification_qubits > self.num_qubits:
            raise ValueError(f"Number of qubits must be at least equal to {self.classification_qubits}")

    def forward(self, state: torch.Tensor, num_layers_to_execute: Optional[int] = None):
        """
        Calculate the probability distribution from quantum state measurements.

        :param num_layers_to_execute: The number of layers to execute.
        :param state: a tensor of shape (batch, 256) to initialize the circuit with the image (16x16).
        :return: probability of being 1.

        State vector have shape (batch, 256). Referring to a single image it has shape (,256), where each of the
        elements corresponds to the amplitude of the system being in one of the 256 = 2^8 states, {|00000000>,
        |00000001>, ..., |11111110>, |11111111>}. The first 128 are referred to states |0XXXXXXX>, meaning that if
        the classes are 2 forward pass return the probabilities to be in one of the first 128 states. If the classes
        are greater than 2, the total number of possible states is calculated with the classification qubits, then
        the 256 states are divided in groups each one associated to a class. If the number of class n is lower than the
        groups only the first n groups are considered.

        Example 1: 3 classes -> 2 classification qubits = 4 possible states |00>, |01>, |10>, |11>, but only the first
        three are considered.
        Example 2: 4 classes -> 2 classification qubits = 4 possible states |00>, |01>, |10>, |11> each one associated
        to a class.
        Example 3: 5 classes -> 3 classification qubits = 8 possible states |000>, |001>, |010>, |011>, |100>, |101>,
        |110>, |111>, |111>, but only the first 5 are considered.

        """
        if self.num_classes == 2:
            state_vector = self.quantum_layer(state=state, num_layers_to_execute=num_layers_to_execute)
            probabilities = torch.sum(torch.abs(state_vector[:, :2 ** (self.quantum_layer.num_qubits - 1)]) ** 2, dim=1)
            return probabilities.type(torch.float32)

        state_vector = self.quantum_layer(state=state, num_layers_to_execute=num_layers_to_execute)
        total_states = 2 ** self.classification_qubits

        probabilities = torch.zeros(state_vector.shape[0], self.num_classes, dtype=torch.float32)

        for idx in range(self.num_classes):
            # Calculate the index range for each class
            start_idx = int(idx * (2 ** self.num_qubits / total_states))
            end_idx = int((idx + 1) * (2 ** self.num_qubits / total_states))
            probabilities[:, idx] = torch.sum(torch.abs(state_vector[:, start_idx:end_idx]) ** 2, dim=1)

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
            loss_function: Optional[torch.nn.modules.loss] = None,
            num_layers_to_execute: Optional[int] = None, show_plot: Optional[bool] = False) -> tuple:
        """
        Train the quantum circuit given a set of training data. The loss function is considered to be the Binary
        Cross Entropy Loss for binary classification, Negative Log Likelihood when all the states are used, meaning
        that the output are probabilities, CrossEntropyLoss when not all the states are used.

        :param show_plot: if true, show the plot of the loss and the accuracy.
        :param dataloader: dataloader with training data.
        :param learning_rate: learning rate of the optimizer.
        :param epochs: number of epochs.
        :param loss_function: loss function.
        :param num_layers_to_execute: The number of layers to execute
        :param show_plot: If True, plot the loss during training.
        :return: tuple containing accuracy history and loss history per epoch.
        """

        if loss_function is None:
            if self.num_classes == 2:
                loss_function = torch.nn.BCELoss()  # binary cross entropy for binary classification

            elif 2 ** self.classification_qubits == self.num_classes:  #output are probabilities
                loss_function = torch.nn.NLLLoss()  # Negative Log Likelihood for multinomial classification

            else:  #outputs can be considered as logits
                loss_function = torch.nn.CrossEntropyLoss()

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
            accuracy_per_epoch = list()
            targets_per_epoch = list()

            with tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}/{epochs}') as tqdm_epoch:

                for _, (data, targets) in tqdm_epoch:

                    data = data / torch.linalg.norm(data, dim=1).view(-1, 1)

                    optimizer.zero_grad()

                    output = self.forward(state=data, num_layers_to_execute=num_layers_to_execute)

                    if self.num_classes == 2:  #binary classification -> BinaryCrossEntropy

                        batch_accuracy = torch.sum((output > 0.5) == targets).item() / len(output)
                        accuracy_per_epoch.append(batch_accuracy)
                        loss = loss_function(output, targets)

                    elif 2 ** self.classification_qubits == self.num_classes:  #output are probabilities->NegativeLogLik

                        predictions = torch.argmax(output, dim=1)
                        batch_accuracy = torch.sum(predictions == targets).item() / len(predictions)
                        accuracy_per_epoch.append(batch_accuracy)
                        log_probs = torch.log(output)
                        loss = loss_function(log_probs, targets)

                    else:  #output can be considered logits -> CrossEntropyLoss

                        predictions = torch.argmax(output, dim=1)
                        batch_accuracy = torch.sum(predictions == targets).item() / len(predictions)
                        accuracy_per_epoch.append(batch_accuracy)
                        loss = loss_function(output, targets)

                    running_loss += loss.item()

                    loss.backward()

                    optimizer.step()

                    tqdm_epoch.set_postfix(loss=loss.item(),
                                           accuracy=batch_accuracy)

            avg_time_per_epoch += time() - t_start

            loss_history.append(running_loss / len(dataloader))

            accuracy.append(sum(accuracy_per_epoch)/len(accuracy_per_epoch))

            # print the time
            print("Time per epoch: ", time() - t_start)

            # print the loss
            print("Epoch: ", epoch + 1, "Loss: ", loss_history[epoch])

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

