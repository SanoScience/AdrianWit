This folder contains SOFA software files (*.scn) to test different scenarios of body physics.

Test using SOFA software

https://www.sofa-framework.org/


Liver_MSH_Liver_Przemek.scn contains the same physical scenario as used to train neural networks in (2.Data_generation). Whenever you change the code in MAIN_Generate_scenarios.py, test the solution here.

Liver_SOFA_test.scn contains the default liver model used by SOFA. This model focuses on testing collision models and barycentric mapping between a higher resolution surface model and a volumetric model.

Liver_MSH_test_2_HyperElastic.scn is used to test different constitutive models, like HyperElastic, based on the Saint Venant–Kirchhoff approach. The scene contains a comparison between linear and nonlinear behavior models.

Read_Diff_Meshes.scn is used to test the properties of different types of meshes: *.msh, *.obj, and *.stl. These common file extensions have different material properties and need to be solved in different ways.


Use these scenarios as a tool to test different behavior models for neural network training.
Make sure to change the path directions to the meshes.

