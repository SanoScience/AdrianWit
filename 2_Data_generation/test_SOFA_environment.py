# Required import for SOFA within python
import Sofa
import numpy as np
import os

# Choose in your script to activate or not the GUI
USE_GUI = True      # True # False


def main():
    import SofaRuntime
    import Sofa.Gui

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

    # Call the SOFA function to create the root node
    root = Sofa.Core.Node("root")
    createScene(root)
    Sofa.Simulation.init(root)

    finalTime = root.time.value
    print(finalTime)

    if USE_GUI:
        Sofa.Gui.GUIManager.Init('myscene', "qglviewer")
        Sofa.Gui.GUIManager.createGUI(root, __file__)
        Sofa.Gui.GUIManager.SetDimension(1500, 1200)

        # Initialization of the scene will be done here
        Sofa.Gui.GUIManager.MainLoop(root)
        Sofa.Gui.GUIManager.closeGUI()
        print("GUI was closed")

    print("Simulation is done.")

# Same createScene function as in the previous case
def createScene(root):

    root.gravity = [0, -9.81, 0]
    root.dt = 0.02
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

    liver.addObject('TetrahedronFEMForceField', name="FEM", youngModulus=2000, poissonRatio=0.4, method="large")
    liver.addObject('MeshMatrixMass', massDensity=1)

    liver.addObject('ConstantForceField', indices="0", forces=[0,0,0])
    liver.addObject('FixedConstraint', indices=[3, 21, 39, 56, 64])

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