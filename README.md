# QML-early_exit
A repositories that contains strategies about how to implement Early-Exit in a Quantum Machine Learning Framework 
## General idea
Simply speaking, the idea is to see if we can develop strategy to execute task using for some exampkles a less amount of layers. This could help the research in the paradigm of NISQ devices, the less we use the quantum part the less the noise. So the first step is to take a well-working baseline model to make classification on MNIST dataset containing only zeros and ones digits, then: 
1. train the whole model
2. 'disconnect' some final layers and attach a new classificator retraining the model (freezing the disconnected layers)
3. repeat 2. with a smaller part of the model
4. look at what happen in the classification results of the three classificator

