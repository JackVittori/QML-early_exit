# QML-early_exit
A repositories that contains strategies about how to implement Early-Exit Quantum Machine Learning. 
## General idea
Simply speaking, the idea is to see if we can develop strategy to execute task using for some exampkles a less amount of layers. This could help the research in the paradigm of NISQ devices, the less we use the quantum part the less the noise. So the first step is to take a well-working baseline model to make classification on MNIST dataset containing only zeros and ones digits, then: 
1. train the whole model
2. 'disconnect' some final layers and attach a new classificator retraining the model (freezing the disconnected layers)
3. repeat 2. with a smaller part of the model
4. look at what happen in the classification results of the three classificator

# Pre-requisites

A **vector space** E is a non empty set in which are defined two operations: 
- a map $E \times E \rightarrow E$ called *sum* that associates a couple of elements of E to a third element in E;
- a map $\mathbb{C} \times E \rightarrow E$ called *scalar multiplication* that associates to a couple composed by a complex number and an element of E an element of E.

In addition, being E a vector space it holds: 
- commutative property of sum;
- associative property of sum;
- associative property of scalar multiplication;
- distributive property, for scalars, of scalar multiplication;
- distributive property, for vectors, of scalar multiplication;
- existence of a scalar neutral element in scalar multiplication;
- existence of a vector neutral element in sum.

Sum is an internal operation in the vectorial space, given that all the partecipating elements are in the same vectorial space, while the scalar multiplication is external because the scalar element is not an element of E but of a set $S \subset \mathbb{C}$. Then E is a vector space on field S, but in the particular cases where $S = \mathbb{C}$ or $S = \mathbb{R}$, E is called a complex or real vector space.

Bra-Ket notation, introduced by Dirac in 1939 to describe quantum states, can be used also for abstract mathematical objects, such as vectors of the vector space E. The word *bracket* means "parenthesis". Denoting by *ket* $` \ket{a} `$ an element of E, *bra* $` \bra{b} `$ denotes an element belonging to the dual space of E. 

**Scalar product** or **inner product** is an operation that associates an ordered couple of elements of E an element of scalar field S on which the vector space E is defined: 

```math
\braket{\cdot | \cdot}\, : \, E \times E \rightarrow S
```

$` \forall \ket{a},\ket{b},\ket{c} \in E, \, \forall \alpha, \beta \in S`$: 
- $` \braket{a|b} = \braket{b|a}^*`$: commutative property does not hold;
- $` \bra{c} (\alpha \ket{a} + \beta \ket{b}) = \alpha \braket{c|a} + \beta \braket{c|b}`$: it is linear wrt ket vector;
- $` \braket{a|a} \geq 0, \braket{a|a} \iff \ket{a} = \ket{0}`$
- if $\braket{a|b} = \braket{b|a} = 0$, they are orthogonal. 
## Qubit representation

For a quantum system of dimension 2, to identify a qubit we can use computational basis $` \{ \ket{0}, \ket{1} \} `$

## Quantum Operators
Quantum states are usually represented by kets $\ket{\psi}$. The vector representation of a single qubit is: 

```math
\ket{\psi} = \alpha \ket{0} + \beta \ket{1} \rightarrow \begin{pmatrix}
\alpha \\
\beta
\end{pmatrix}
```

The computational basis vector representation is $` \ket{0} = \begin{pmatrix}
1 \\
0
\end{pmatrix}, \ket{1} = \begin{pmatrix}
0 \\
1
\end{pmatrix}`$. Quantum gates, acting on $n$ qubits, are represented by unitary matrix $2^n \times 2^n$. The set of all gates with the group operation of matrix multiplication is the unitary group $U(2^n)$.


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

Data is encoded in the amplitude of quantum states. A normalized classifcal $N$ dimensional datapoint $x$ is represented by the amplitudes of a $n$-qubit quantum state $\ket{\phi_x}$ as: 

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


### RX

It represents a single qubit X rotation: 

```math
R_x(\phi) = e^{- i \phi \sigma_x /2} = 
\begin{pmatrix}
\cos(\phi /2) & -i \sin (\phi /2)  \\
-i \sin (\phi /2) & \cos (\phi /2)
\end{pmatrix}
```

### RY

It represents a single qubit Y rotation: 

```math
R_y(\phi) = e^{- i \phi \sigma_y /2} = 
\begin{pmatrix}
\cos(\phi /2) & - \sin (\phi /2)  \\
\sin (\phi /2) & \cos (\phi /2)
\end{pmatrix}
```

### RZ

It represents a single qubit Z rotation: 
```math
R_z(\phi) = e^{- i \phi \sigma_z /2} = 
\begin{pmatrix}
e^{- i \phi /2} & 0 \\
0 & e^{ i \phi /2}
\end{pmatrix}
```
### CNOT

### AmplitudeDamping

AmplitudeDamping is a single qubit opeator that can be used to model interaction with the environment, leading to changes in the state populations of a qubit. This is the phenomenon behind scattering, dissipation, attenuation and spontaneous emission. This can be expressed with the following Kraus matrices where $\gamma \in [0,1]$ is the amplitude damping probability: 

```math
K_0 = 
\begin{pmatrix}
1 & 0 \\
0 & \sqrt{1-\gamma}
\end{pmatrix}
\\
K_1 = 
\begin{pmatrix}
0 & \sqrt{\gamma} \\
0 & 0
\end{pmatrix}
```


  

