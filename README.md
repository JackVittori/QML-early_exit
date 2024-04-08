# QML-early_exit
A repositories that contains strategies about how to implement Early-Exit in a Quantum Machine Learning Framework 
## General idea
Simply speaking, the idea is to see if we can develop strategy to execute task using for some exampkles a less amount of layers. This could help the research in the paradigm of NISQ devices, the less we use the quantum part the less the noise. So the first step is to take a well-working baseline model to make classification on MNIST dataset containing only zeros and ones digits, then: 
1. train the whole model
2. 'disconnect' some final layers and attach a new classificator retraining the model (freezing the disconnected layers)
3. repeat 2. with a smaller part of the model
4. look at what happen in the classification results of the three classificator

# PennyLane

##Wires 
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

## QubitStateVector

QubitStateVector is involved in State Preparation step that prepares the subsystem given the ket vector (state) in the computational basis. The *ket vector* is a `array[complex]` of size $2\cdot \text{len}(\text{wires})$.


  

