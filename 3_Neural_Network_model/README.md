This folder contains a Neural Network prediction (regression) model written in Python scripts.

Two network models have been created: Scrach_NN_learning.py and PyTorch_learning.py

Scrach_NN_learning.py contains a neural network design in which all functions have been elaborated in the form of classes. However, this model has not yet been fully tested for its applicability.

PyTorch_learning.py contains a model built based on PyTorch libraries. This is currently the main model in use.

The PyTorch_learning.py script retrieves the generated batch data and uses it to train the neural network. The subroutine Create_batches.py creates training batches with the following structure: the first three columns contain the initial position of the model in the x, y, z axes. The last three columns are filled with zeros, in which one of the rows contains information about the value of the incremental displacement of a selected node over the time dt. As an example, a single batch is presented in the file Test_batch.txt. The row number where the value is found corresponds to the node number to which the force was applied during the generation of training data. The values in this row correspond to the displacement values caused by force F for the specific model

Network parameters should be defined in the class module Net(nn.Module). The result of the network training is a file network_PyTorch.pkl containing the parameters of the trained network. This file can be used for predicting the deformation values of the model.

An example of transitioning the network to prediction mode is presented in the file PyTorch_prediction.py


