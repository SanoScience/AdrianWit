import numpy as np
import vtk

def load_init_positions():
    file_path = 'C:/...../Mesh/liver2.npz'
    liver2_data = np.load(file_path)
    return liver2_data['node']

def load_init_elements():
    file_path = 'C:/......../Mesh/liver2.npz'
    liver2_data = np.load(file_path)
    return liver2_data['elem']


class MouseInteractorHighLightActor(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self, parent=None):
        self.AddObserver("LeftButtonPressEvent", self.leftButtonPressEvent)
        self.AddObserver("MouseMoveEvent", self.mouseMoveEvent)
        self.LastPickedActor = None
        self.LastPickedProperty = vtk.vtkProperty()
        self.picked_position = None

    def leftButtonPressEvent(self, obj, event):
        click_pos = self.GetInteractor().GetEventPosition()
        picker = vtk.vtkPropPicker()
        picker.Pick(click_pos[0], click_pos[1], 0, self.GetDefaultRenderer())

        # Get the new actor
        self.NewPickedActor = picker.GetActor()

        # If something was selected
        if self.NewPickedActor:
            # If we picked something before, reset its property
            if self.LastPickedActor:
                self.LastPickedActor.GetProperty().DeepCopy(self.LastPickedProperty)

            # Save the property of the picked actor so that we can restore it next time
            self.LastPickedProperty.DeepCopy(self.NewPickedActor.GetProperty())
            # Highlight the picked actor by changing its properties
            self.NewPickedActor.GetProperty().SetColor(1.0, 0.0, 0.0)
            self.NewPickedActor.GetProperty().SetDiffuse(1.0)
            self.NewPickedActor.GetProperty().SetSpecular(0.0)

            # Save the last picked actor
            self.LastPickedActor = self.NewPickedActor

        self.OnLeftButtonDown()
        return

    def mouseMoveEvent(self, obj, event):
        if self.LastPickedActor:
            click_pos = self.GetInteractor().GetEventPosition()
            picker = vtk.vtkPropPicker()
            picker.Pick(click_pos[0], click_pos[1], 0, self.GetDefaultRenderer())

            world_pos = picker.GetPickPosition()
            self.LastPickedActor.SetPosition(world_pos)

        self.OnMouseMove()
        return


def main():
    # Create a renderer, render window, and render window interactor
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1600, 900)  # Set the window size

    screen_width = render_window.GetScreenSize()[0]
    screen_height = render_window.GetScreenSize()[1]
    x_position = (screen_width - 1600) // 2
    y_position = (screen_height - 900) // 2
    render_window.SetPosition(x_position, y_position)  # Set window position

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Load the initial positions and elements
    init_positions = load_init_positions()
    init_elements = load_init_elements()

    # Create a sphere source for rendering nodes
    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetRadius(0.02)  # Adjust the sphere radius as needed (e.g., 0.05 for smaller spheres)
    sphere_mapper = vtk.vtkPolyDataMapper()
    sphere_mapper.SetInputConnection(sphere_source.GetOutputPort())

    # Create actors for the spheres and add them to the renderer
    for position in init_positions:
        sphere_actor = vtk.vtkActor()
        sphere_actor.SetMapper(sphere_mapper)
        sphere_actor.SetPosition(position[0], position[1], position[2])
        renderer.AddActor(sphere_actor)

    # Set custom interactor style
    style = MouseInteractorHighLightActor()
    style.SetDefaultRenderer(renderer)
    interactor.SetInteractorStyle(style)

    # Initialize and start the visualization
    render_window.Render()
    interactor.Initialize()
    interactor.Start()


if __name__ == "__main__":
    main()