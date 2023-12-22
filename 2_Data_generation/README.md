This folder contains Python scripts whose main purpose is to generate training data for the neural network contained in the folder (3_Neural_Network_model).

To use this code, it is necessary to import the SOFA libraries in Python. Introduction to implementation can be found on the website SOFA Python 3 Documentation

https://sofapython3.readthedocs.io/en/latest/index.html 

The folder contains:

test_SOFA_environment.py is used to test the implementation of SOFA functions and classes in the Python environment. This code invokes the SOFA physics engine as well as the SOFA visualization engine, which can be enabled with the function USE_GUI = False.

The Liver2.npz file refers to Liver_Przemek.msh in the (Meshes) folder. It contains the initial node positions of the model, element indices, vertices, and more. Read it as Numpy matrix.

Main code:

MAIN_Generate_scenarios.py is used for generating various load scenarios on the model using the SOFA physics engine. This script generates different scenarios, each containing a simulation of body deformation under the influence of a 'random_force' vector with random value and direction, which is applied to a randomly selected node 'random_node'. The simulation is then conducted for a specified number of iterations. The number of scenarios and iterations is determined by the user through the parameters 'scenarios_range' and 'iter_nr'. A condition for model stability is the application of boundary conditions in the form of constraints, where the user specifies the node numbers to be constrained in the form of a 'Fixed_Constraint' vector.

Code results:

The result of the code execution is a set of *.txt files saved in the directories of the folder (Resulats_sim). Data about the calculation of body deformation as a result of the simulation can be found in the directory (Resulats_sim\position). Each file contains the simulation results for each scenario for each iteration. The interpretation of the file name is as follows: {scenario_nr}_{iteration}. Each file consists of 6 columns, where the first three correspond to the initial position of all model nodes in the x, y, z axes, respectively, while the last three contain the final position of all model nodes in that iteration. For example, the file 264_11.txt contains the simulation of the model deformation for the 264th scenario in the 11th iteration. Information about selected nodes and the force vector values for each scenario can be found in the directory (Resulats_sim\forces). An example file, f_264.txt, contains information in the order {random_node, random_force_x, random_force_y, random_force_z}.

This data will be used to create batches as input for neural network training. Make sure to change the path directions !!!