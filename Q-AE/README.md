## Quantum AE
From [this](https://fullstackquantumcomputation.tech/blog/quantum-autoencoder/) a quantum state autoencoder is a circuit that takes a statevector as input and it outputs a reduced version of that statevector. And to get the original statevector (approximately) from the encoded statevector we can apply the opposite of a QAE, that is, a Quantum state decoder. This is a great idea because as you might know we don’t get access to a lot of resources in NISQ machines so by applying a QAE we can reduce the use of those resources.

<p align="center">
<img src= https://fullstackquantumcomputation.tech/assets/images/autoencoder/autoencoder.png style="width:50%;"/>
 </p>

In the image there is 4×4 statevector that we want to encode into a 2×2 statevector. We can do this by applying the autoencoder to our circuit. One thing worth noticing is that two qubits were set to 0 state in the process. Nonetheless, there wasn’t a loss of information because we encoded that information inside the two last qubits. Now, if we want to have the original 4×4 statevector we need to apply the decoder to our circuit. Notice that we have to include the qubits that were set in the 0 state.

The general process to construct a quantum autoencoder is:

1. Generate a statevector that we want to reduce.
2. Create an anzast for a set of parameters (this is represented in the image as the AUTOENCODER).
3. Create the decoder by finding the inverse of the AUTOENCODER anzast.
4. Get the cost of the AUTOENCODER by comparing the original state with the restored state from the DECODER.
5. Optimize the parameters by using a classical optimization routine (ADAM, ADA, Stochastic gradient descent, etc..).
