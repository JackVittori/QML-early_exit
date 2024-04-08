# QML-early_exit
A repositories that contains strategies about how to implement Early-Exit in a Quantum Machine Learning Framework 
## General idea
Simply speaking, the idea is to see if we can develop strategy to execute task using for some exampkles a less amount of layers. This could help the research in the paradigm of NISQ devices, the less we use the quantum part the less the noise. So the first step is to take a well-working baseline model to make classification on MNIST dataset containing only zeros and ones digits, then: 
1. train the whole model
2. 'disconnect' some final layers and attach a new classificator retraining the model (freezing the disconnected layers)
3. repeat 2. with a smaller part of the model
4. look at what happen in the classification results of the three classificator

# PennyLane

## Wires 
PennyLane uses the term *wire* to refer to a quantum subsystem.

## Device
This function is used to load a particular quantum device, which can then be used to construct QNodes. In PennyLane a device represents the environment where quantum circuit are executed, that can be a simulator or a real quantum processor.
- **'default.qubit'** is a simple state simulator. A state simulator is used to calculate the time evolution of a quantum system described by a pure state.

- **'default.mixed'** is a mixed-state simulator. A mixed states simulator consider in the evolution the presence of mixed states, results of partial measurements or of the initial uncertainty on the system states ensemble.

## qml.qnode

  *qnode* is an object used to represent a quantum node in the hybrid computational graph. It contains a quantum function (quantum circuit function), corresponding to a variational circuit and the computational device where it is executed. It corresponds to construct a *QuantumTape* instance representing the circuit.
The parameter *interface* is used for the classical backpropagation and it is possible to use torch, tf, jax, autograd etc.
Usually it is used as decorator such as

```python
@qml.qnode(dev, interface="torch")
```

## State Preparation

### QubitStateVector

QubitStateVector is involved prepares the subsystem given the ket vector (state) in the computational basis. The *ket vector* is a `array[complex]` of size $2\cdot \text{len}(\text{wires})$.

### Quantum embedding

A quantum embedding represents classical data as quantum states in a Hilbert space via a quantum feature map.

A *feature map* $\phi$ is a function that acts $\phi: \mathcal{X} \rightarrow \mathcal{F}$ where $\mathcal{F}$ is the feature space. The outputs are called feature vectors.

A *quantum feature map* is a feature map where the vector space $\mathcal{F}$ is a Hilbert space and the feature vectors are quantum states: 

$$
x \rightarrow \ket{\phi(x)}
$$

Typically in variational circuit the map transforms input data through a Unitary Transformation $U_\phi(x)$ whose params depend on input data.

A **quantum embedding** takes classical datapoints translating them into a set of gate parameters in a quantum circuit which outputs a quantum state $\ket{\phi_x}$.

### Basis Embedding

It associates each input with a computational basis state of a qubit system, thus one bit of classical information is represented by one quantum subsystem. Considering input data in a binary string format $x = 1010$ is represented by the 4-qubit quantm state $\ket{1010}$.

Considering a classical dataset $\mathcal{D}$ composed by $x^{(m)} = (b_1,\dots, b_N)$ with $b_i \in \{0,1\}$ where $m = 1,\dots, M$: 

$$
\ket{\mathcal{D}}=\frac{1}{\sqrt{M}} \sum \limits_{m=1}^M \ket{x^{(m)}}
$$

For $N$ bits there are $2^N$ possible basis states. Given $M \ll 2^N$ the basis embedding of $\mathcal{D}$ will be sparse.

Example: $x^{(1)} = 01$ and $x^{(2)}= 11$, the resulting basis encoding

$$
\ket{\mathcal{D}} = \frac{1}{\sqrt{2}} \ket{01} + \frac{1}{\sqrt{2}} \ket{11}
$$

### Amplitude Embedding

Data is encoded in the amplitude of quantum states. A normalized classifcal $N$ dimensional datapoint $x$ is represented byt the amplitudes of a $n$-qubit quantum state $\ket{\phi_x}$ as: 

$$
\ket{\phi_x}=\sum \limits_{i=1}^N x_i \ket{i}
$$

where $N=2^n$, $x_i$ is the i-th element of $x$ and $\ket{i}$ is the i-th computational basis state. 
Considering the concatenation of all the input examples $x^{(m)}$ of a classical dataset $\mathcal{D}$

$$
\alpha = C_{\text{norm}} x_1^{(1)}, \dots, x_N^{(1)}, x_1^{(2)}, \dots, x_N^{(2)}, x_1^{(M)}, \dots, x_N^{(M)}
$$

where $C_{\text{norm}}$ is the normalization constant. Considering that $|\alpha|^2 = 1$, the input dataset can be represented in the computational basis as: 

$$
\ket{\mathcal{D}} = \sum \limits_{i=1}^{2^n} \alpha_i \ket{i}
$$

where $\alpha_i$ are the elements of the amplitude vector $\alpha$ and $\ket{i}$ the computational basis state. 

The number of amplitudes to be encoded is $N \times M$, a system of $n$ qubits provides $2^n$ amplitudes, so amplitude embedding requires $n \geq \log_2 (NM)$ qubits.

  

