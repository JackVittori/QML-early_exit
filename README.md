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
- a map: $E \times E \rightarrow E$ called *sum* that associates a couple of elements of E to a third element in E;
- a map: $\mathbb{C} \times E \rightarrow E$ called *scalar multiplication* that associates to a couple composed by a complex number and an element of E an element of E.

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
- $\braket{a|b} = \braket{b|a} = 0 \iff \ket{a},\ket{b} \in E$ are orthogonal.

$` \forall \ket{a},\ket{b}, \ket{c} \in E \, \forall \alpha, \beta \in S `$, defining $` \ket{s} = \alpha \ket{a} + \beta \ket{b}`$ the product $` \braket{c|s} = \alpha \braket{c|a} + \beta \braket{c|b}`$, but $` \braket{s|c} = \braket{c|s}^* = \alpha^* \braket{a|c} + \beta^* \braket{b|c}`$, meaning that is *antilinear* in $\bra{s}$. In order to solve this sort of asimmetry, it can be defined a **dual** vector space $`E^*`$ containing the element of E,  but in bra form. In this way it exists a one-to-one correspondence between the elements of E and the elements of $`E^*`$, $` \forall \ket{a} \in E \exists ! \bra{a} \in E^* `$ and viceversa. The scalar product is now defined as: 

```math
\braket{\cdot | \cdot}: E^* \times E \rightarrow \mathbb{C}
```
and it becomes linear with respect to bra and ket. In general $`\bra{a} \in E^*`$ is said to be the dual vector of $`\ket{a} \in E`$ and if $`\ket{a} = \alpha \ket{b}`$ the dual is $`\bra{a} = \alpha^* \bra{b}`$.

The **norm** in a vector space E is an application $`|| \cdot ||: \, E \rightarrow \mathbb{R}`$. A vector space where is defined a norm is said to be a **metric space** or a **normed space** and it is identified by the couple $`(E, ||\cdot ||)`$. In a normed space E it is possible to define a norm using the scalar product, it is said to be that the norm is induced by the scalar product. Given the fact that the norm is only defined for vectors now, it is sufficient to indicate the vector using only the letter, thus the norm of vector $`\ket{a} \in E`$ can be defined as:

```math
\begin{align}
 ||a|| = \sqrt{\braket{a|a}} \\
 || \alpha \ket{a} || = \sqrt{|\alpha|^2 \braket{a|a}} = |\alpha| ||a||
\end{align}
```

Given the succession $`\{ \ket{a_k}\}_{k=1}^{\infty}`$ in the normed space $`(E,||\cdot||)`$, it **converges in norm** to a vector $`\ket{a} \in E \iff \forall \epsilon > 0, \exists k_0(\epsilon)`$ such that $`||a_k - a || < \epsilon`$, $`\forall k \geq k_0`$. It can be written as $`\underset{k \rightarrow \infty}{\lim} \ket{a_k} = \ket{a}`$. 

The succession $`\{ \ket{a_k}\}_{k=1}^{\infty}`$ in the normed space $`(E,||\cdot||)`$ is said to be of **Cauchy** $` \iff \forall \epsilon > 0, \exists k_0(\epsilon)`$ such that $` ||a_k - a_m || < \epsilon`$, $` \forall k,m \geq k_0`$. It can be proved that every convergent succession is a Cauchy succession, but not convergent Caucy succession can exist. 

A vector space is said to be **complete** if and only if every Caucy succession of his elements, given a norm, converges to an element belonging to the same vectorial space. A metric space that is normed and complete is said to be **Banach space**. If the norm that induces the convergence and the completeness is induced by the scalar product $` || \cdot || \sqrt{\braket{\cdot|\cdot}}`$ the complete vectorial space is an **Hilbert space**.

The concept of **operator** generalizes in a vector space a function that is instead definde in a scalar field. Considering an operation $f$ in the vector space $E$ that associates to $`\ket{a} \in D \subset E`$ a vector $`\ket{b} \in E`$, $`f: D \rightarrow E`$. In operatorial form the relationship can be written as: 

```math
\hat{F} \ket{a} = \ket{b}
```
An operator is defined if and only if is defined is *action*. In general the product of two operators is not commutative but it is if the so called **commutator** between them is null: $`[\hat{F}, \hat{G}] = \hat{F} \hat{G} - \hat{G} \hat{F}`$.
In a vector space $E$ the N vectors of the set $`\{ ket{e_k}\}_{k=1}^N \subset E`$, that does not contain the null vector, are linear independent if and only if $`\sum \limits_{k=1}{N} c_k \ket{e_k} = \ket{0}`$ is equal to the condition $`c_k = 0 \, \forall k \in \{1,2,\dots,N \}`$. The maximum number of linear independent vectors is equal to the dimension of the vector space. 

Given $`\{ \ket{e_k}\}_{k=1}^N \subset E`$ is a set of linear independent vectors, this set is a **basis** if and only if $`\forall \ket{a} \in E`$ it is possible to write: 

```math
\ket{a} = \sum \limits_{k=1}^N a_k \ket{e_k}
```
Furthermore, the $`a_k`$ coefficients are unique.

A linear operator $`\hat{F}:D \rightarrow R`$ is univoquely defined by its action on the vectors of a basis $`\{ ket{e_k}\}_{k=1}^N \subset D`$. Once the set $`\{ \ket{f_k} = \hat{F} \ket{e_k}\}_{k=1}^N \in R`$ is known, the action for any vector is the defined and: 

```math
\hat{F} \ket{a} = \hat{F} \sum \limits_{k=1}^N a_k \ket{e_k} = \sum \limits_{k=1}^N a_k \hat{F} \ket{e_k} = \sum \limits_{k=1}^N a_k \ket{f_k} \in R.
```

Denoting by $`\hat{F}:E \rightarrow E`$ a linear operator, the **Hermitian adjoint**  $`\hat{F}^{\dagger}`$ of $`\hat{F}`$ is the operator whose action $`\bra{a} \hat{F}^{\dagger}`$ represents the dual of the vector $`\hat{F}\ket{a}`$. Another definition in terms of the scalar product is $`\forall \ket{a},\ket{b} \in E \, , \bra{a}, \bra{b} \in E^*`$ and $`\hat{F}: E \rightarrow E`$: 

```math
\braket{a|\hat{F}|b} = \braket{b|\hat{F}|a}^*
```

A linear operator $`\hat{H}: E \rightarrow E`$ is **Hermitian** if $`\hat{H} = \hat{H}^{\dagger}`$. They are important because the eigenvalues of Hermitian operators are real and eigenvectors relative to different eigenvalues are orthogonal.
An invertible operator $`\hat{U}:E \rightarrow E`$ is **unitary** if and only if $`\hat{U}^{-1} = \hat{U}^{\dagger}`$. Unitary operators keep the scalar operator and in Hilbert spaces they are called *isometric* because they keep the norm.
Eigenvalues of unitary operators are complex values of unitary module, that is *pure phases*.

Considering a linear operator $`\hat{A}: E_N \rightarrow E_N`$ and a basis $`\{ \ket{e_k} \}_{k=1}^N`$ of $`E_N`$ the vector obtained applying $`\hat{A}`$ on the j-th vector of the basis has the following representation: 

```math
\hat{A}\ket{e_j} = A_j^k \ket{e_k}
```
where the coefficient $`A_j^k`$ with $`j,k \in \{1,2,\dots, N \}`$ is the k-th component with respect to basis of the vector $`\hat{A} \ket{e_j}`$. The pedix represent the vector of the basis where the operator has been applied while the apex represent the component. With this decomposition there are $`N^2`$ scalars that univoquely defined the action in the vectorial space $`E_N`$, i.e. the operator. In **matrix notation** the operator can be represented with respect to the basis $`\{ \ket{e_k} \}_{k=1}^N`$ by a $`N \times N`$ matrix with the apex representing the row and the pedix representing the column. 
In the case of orthonormal basis $`\{ \ket{u_k}\}_{k=1}^N`$, $`\braket{u_k|u_j} = \delta_{kj}`$ and the matrix representation of the operator has the easiest form, written by the sandwich $`A^k_j = \braket{u_k | \hat{A}|u_j}`$.

If the matrix identity that represents the vectorial $`\hat{A}\ket{a} = \ket{b}`$ is $`Aa = b`$, then for the dual $`\bra{b} = \bra{a}\hat{A}^{\dagger}`$ is represented by $`b^{\dagger} = a^{\dagger} A^{\dagger}`$.

In Quantum Mechanics all the property of a physics system that can be measured, assciating to them a real quantity, are defined as **observables**. The observables are represented by hermitian operators acting on Hilbert spaces. Denoting as $`\hat{H}: E_N \rightarrow E_N`$ an hermitian operator and $`\{ \ket{\phi_k} \}_{k=1}^N`$  an orthonormal basis of $`E_N`$ composed by eigenvectors of $`\hat{H}`$ with eigenvalues $`\{ \lambda_k\}_{k=1}^N`$, the eigenvalues are the unique possible results of measuring the physical quantity associated to the operator $`\hat{H}`$ on a given physics system.

If a physics system is in a state described by the vector $`\ket{\psi} \in E_N`$ the **probability** to obtain $`\lambda_k`$ as results of measuring the physical quantity to which the operator $`\hat{H}`$ is associated is given by $`c^k = \braket{\phi_k|\psi}`$. This means to make the system collapse in the eigenstate $`\ket{\phi_k}`$. 

It is called **expectation value** of an observable $`\hat{H}`$ in a state $`\ket{\phi}`$ the quantity: 

```math
\left\langle \hat{H} \right\rangle_{\psi} = \braket{\psi|\hat{H}|\psi} = \braket{\phi_j|\hat{H}|\phi_k} c^{j*}c^k = \braket{\phi_j|\phi_k} c^{j*} c^k \lambda_k = \sum \limits_{k=1}^N |c^k|^2 \lambda_k 
```

The *root mean squared error* of an observable $`\hat{H}`$ in a state $`\ket{\psi}`$ is defined as

```math
\Delta H_\psi = \sqrt{\left \langle (\hat{H} - \left\langle \hat{H} \right\rangle_{\psi} \hat{I})^2 \right \rangle_{\psi}}
```

That is null only if $`\ket{\psi}`$ is an eigenstate of $`\hat{H}`$.
*Side note: This definition can be used to enunciate the Indetermination Principle of Heisenberg that states that given two (limited) hermitian operators $`\hat{A}, \hat{B}: E_N \rightarrow E_N`$, $`\forall \ket{\psi} \in E_N`$ it holds $`\Delta A_{\psi}\Delta B_{\psi} \geq \frac{1}{2} |\left \langle [ \hat{A},\hat{B}] \right \rangle_{\psi}|`$.*

# Qubit
A qubit is a quantum system of dimension two and is usually represented using $\ket{\psi}$. To identify a qubit we can use the so called *computational basis* $` \{ \ket{0}, \ket{1} \} `$ living in the Hilbert space $\mathcal{H}, dim(\mathcal{H}) = 2$
The vector representation of a single qubit is: 

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
\end{pmatrix}`$.

Quantum gates, acting on $n$ qubits, are represented by unitary matrix $2^n \times 2^n$. The set of all gates with the group operation of matrix multiplication is the unitary group $U(2^n)$.

## RX

It represents a single qubit X rotation: 

```math
R_x(\phi) = e^{- i \phi \sigma_x /2} = 
\begin{pmatrix}
\cos(\phi /2) & -i \sin (\phi /2)  \\
-i \sin (\phi /2) & \cos (\phi /2)
\end{pmatrix}
```

## RY

It represents a single qubit Y rotation: 

```math
R_y(\phi) = e^{- i \phi \sigma_y /2} = 
\begin{pmatrix}
\cos(\phi /2) & - \sin (\phi /2)  \\
\sin (\phi /2) & \cos (\phi /2)
\end{pmatrix}
```

## RZ

It represents a single qubit Z rotation: 
```math
R_z(\phi) = e^{- i \phi \sigma_z /2} = 
\begin{pmatrix}
e^{- i \phi /2} & 0 \\
0 & e^{ i \phi /2}
\end{pmatrix}
```
## CNOT

## Quantum embedding

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


  

