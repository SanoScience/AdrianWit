# Required import for SOFA within python
import Sofa
import numpy as np
import os
import random
import copy

# Choose in your script to activate or not the GUI
USE_GUI = False      # True # False
scenarios_range = 500
iter_nr = 20
Fixed_Constraint = [406, 409, 410, 413, 329, 349, 348, 352, 353]

# Save config file
config_file_path = 'C:/Users/Sharkoon/Documents/SOFA_Python_Project/SOFA_gene_config.txt'
with open(config_file_path, 'w') as file:
    file.write(f"scenarios_nr = {scenarios_range} \niter_nr = {iter_nr}")

# Function to check and clear the folder
def clear_folder(folder_path):
    if os.listdir(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

# Clear the folder at the beginning of the script if not empty
folder_path_pos = 'C:/Users/Sharkoon/Documents/SOFA_Python_Project/Resulats_sim/position/'
clear_folder(folder_path_pos)


def main():
    import SofaRuntime
    import Sofa.Gui

    for scenario_nr in range(scenarios_range):  # Define the scenario generating loop

        # Load SOFA plugins
        SofaRuntime.importPlugin("MultiThreading")
        SofaRuntime.importPlugin("Sofa.Component.Collision.Detection.Algorithm")
        SofaRuntime.importPlugin("Sofa.Component.Collision.Detection.Intersection")
        SofaRuntime.importPlugin("Sofa.Component.Collision.Geometry")
        SofaRuntime.importPlugin("Sofa.Component.Collision.Response.Contact")
        SofaRuntime.importPlugin("Sofa.Component.Constraint.Projective")
        SofaRuntime.importPlugin("Sofa.Component.IO.Mesh")
        SofaRuntime.importPlugin("Sofa.Component.LinearSolver.Iterative")
        SofaRuntime.importPlugin("Sofa.Component.Mapping.Linear")
        SofaRuntime.importPlugin("Sofa.Component.Mass")
        SofaRuntime.importPlugin("Sofa.Component.MechanicalLoad")
        SofaRuntime.importPlugin("Sofa.Component.ODESolver.Backward")
        SofaRuntime.importPlugin("Sofa.Component.SolidMechanics.FEM.Elastic")
        SofaRuntime.importPlugin("Sofa.Component.StateContainer")
        SofaRuntime.importPlugin("Sofa.Component.Topology.Container.Constant")
        SofaRuntime.importPlugin("Sofa.Component.Topology.Container.Dynamic")
        SofaRuntime.importPlugin("Sofa.Component.Visual")
        SofaRuntime.importPlugin("Sofa.GL.Component.Rendering3D")
        SofaRuntime.importPlugin("Sofa.Component.SolidMechanics.FEM.HyperElastic")

        #Call the SOFA function to create the root node
        root = Sofa.Core.Node("root")

        gravity_x = 0
        gravity_y = 0
        gravity_z = 0

        # Save gravity vector
        # gravity_vector = np.array([gravity_x, gravity_y, gravity_z])
        # gravity_file_path = f"C:/Users/Sharkoon/Documents/SOFA_Python_Project/Resulats_sim/gravity/g_{i}.txt"
        # np.savetxt(gravity_file_path, gravity_vector, fmt='%.2f')

        random_node = random.randint(0, 544)
        random_force1 = round(random.uniform(-10, 10), 3)
        random_force2 = round(random.uniform(-10, 10), 3)
        random_force3 = round(random.uniform(-10, 10), 3)

        # Save forces info
        output_path_force = f'C:/Users/Sharkoon/Documents/SOFA_Python_Project/Resulats_sim/forces/f_{scenario_nr}.txt'
        # Create a structured array with different data types
        force_data = np.array(
            [(random_node, random_force1, random_force2, random_force3)],
            dtype=[('node', 'i'), ('force1', 'f8'), ('force2', 'f8'), ('force3', 'f8')]
        )
        np.savetxt(output_path_force, force_data, delimiter='\t', fmt=['%i', '%.8f', '%.8f', '%.8f'])

        createScene(root, gravity_x, gravity_y, gravity_z, 0.01, random_node, random_force1, random_force2, random_force3)
        Sofa.Simulation.init(root)

        # SAVE BC to FILES
        output_path_BC_nodes = f'C:/Users/Sharkoon/Documents/SOFA_Python_Project/Resulats_sim/BC_Boundary/Nodes_Fix_nr.txt'
        # check if file still exist
        if os.path.exists("output_path_BC_nodes"):
            os.remove("output_path_BC_nodes")

        formatter = np.vectorize('{:.0f}'.format)  # make format .0f
        # save fixed nodes ind to file
        np.savetxt(output_path_BC_nodes, formatter(Fixed_Constraint), delimiter='\t', fmt='%s')

        # Run the simulation for (iteration) steps
        for iteration in range(iter_nr):
            print(f'Scenario: {scenario_nr}, Iteration: {iteration}')

            input_liver_pos = copy.deepcopy(root.liver.collision.Store_Forces.position.value)
            #print(f'Input_pos: {input_liver_pos}')
            Sofa.Simulation.animate(root, root.dt.value)
            output_liver_pos = root.liver.collision.Store_Forces.position.value
            #print(f'Output_pos: {output_liver_pos}')

            # Combine input and output positions
            inp_out_pos = np.hstack((input_liver_pos, output_liver_pos))
            #print(f'combined_pos: {inp_out_pos}')

            # Save results to file
            position_iter = f'{scenario_nr}_{iteration}'
            output_path_position = f'C:/Users/Sharkoon/Documents/SOFA_Python_Project/Resulats_sim/position/{position_iter}.txt'
            # Format and save to file
            np.savetxt(output_path_position, inp_out_pos, delimiter='\t', fmt='%.8f')

        # finalTime = root.time.value
        # print(finalTime)

        if USE_GUI:
            Sofa.Gui.GUIManager.Init('myscene', "qglviewer")
            Sofa.Gui.GUIManager.createGUI(root, __file__)
            Sofa.Gui.GUIManager.SetDimension(1500, 1200)

            # Initialization of the scene will be done here
            Sofa.Gui.GUIManager.MainLoop(root)
            Sofa.Gui.GUIManager.closeGUI()
            print("GUI was closed")

        print("Simulation is done.")
        root = None  # This should clear the root node and prepare for the next iteration


def createScene(root, gravity_x, gravity_y, gravity_z, dt, random_node, random_force1, random_force2, random_force3):

    root.gravity = [gravity_x, gravity_y, gravity_z]
    dt = 0.02 if dt is None else dt
    root.dt = dt
    root.addObject('DefaultVisualManagerLoop')
    root.addObject('DefaultAnimationLoop')

    # Collision pipeline
    root.addObject('CollisionResponse')  # CollisionPipeline
    root.addObject('BruteForceBroadPhase', name="N2")
    root.addObject('BVHNarrowPhase')
    root.addObject('MinProximityIntersection', name="Proximity", alarmDistance=0.5, contactDistance=0.2)
    root.addObject('DefaultContactManager', name="Response", response='PenalityContactForceField')

    # Load mesh file
    root.addObject('MeshGmshLoader', name="Liver_Przemek", filename="C:/Users/Sharkoon/Dropbox/SANO/Meshes/Liver_Przemek.msh")

    # Add LIVER child node
    liver = root.addChild('liver')
    liver.addObject('EulerImplicitSolver', name="EulerImplicit",  rayleighStiffness="0.1", rayleighMass="0.1")
    liver.addObject('CGLinearSolver', name="CG Solver", iterations=50, tolerance=1e-09, threshold=1e-09)

    liver.addObject('MechanicalObject', name="Liver_DOFs", src="@../Liver_Przemek")
    liver.addObject('TetrahedronSetTopologyContainer', name="topo", src="@../Liver_Przemek")
    liver.addObject('TetrahedronSetGeometryAlgorithms', template="Vec3", name="GeomAlgo", drawTetrahedra=1)

    liver.addObject('TetrahedronFEMForceField', name="FEM", youngModulus=5000, poissonRatio=0.3)
    liver.addObject('MeshMatrixMass', massDensity=1)

    # Add Nodal Force Field !!!
    liver.addObject('ConstantForceField', indices=random_node, forces=[random_force1, random_force2, random_force3])

    # Add Fixed_Constraint !!!
    liver.addObject('FixedConstraint', indices=Fixed_Constraint)

    # Add VISUAL child x2 node
    visual = liver.addChild('Visual')
    visual.addObject('OglModel', src="@../../Liver_Przemek", dx="0")
    visual.addObject('BarycentricMapping', name="visual mapping", input="@../Liver_DOFs")

    # Add COLLISIONS child x2 node
    collision = liver.addChild('collision')
    collision.addObject('MeshTopology', name="mesh_collision_fine", src="@../../Liver_Przemek")
    collision.addObject('MechanicalObject', name="Store_Forces", scale="1.0")
    collision.addObject('TriangleCollisionModel', name="CollisionModel", contactStiffness="3", selfCollision="0", group="1")
    collision.addObject('BarycentricMapping', name="CollisionMapping", input="@../Liver_DOFs", output="@Store_Forces")

    return root

# Function used only if this script is called from a python environment
if __name__ == '__main__':
    main()