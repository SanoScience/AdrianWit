import matplotlib.pyplot as plt
import numpy as np

elem_on = 0
selected_node = None
positions = None  # Global variable for node positions
scatter = None  # Global variable for scatter plot


def load_init_positions():
    file_path = 'C:/Users/...../Mesh/liver2.npz'
    liver2_data = np.load(file_path)
    return liver2_data['node']


def load_init_elements():
    file_path = 'C:/Users/......../Mesh/liver2.npz'
    liver2_data = np.load(file_path)
    return liver2_data['elem']


def load_fixed_nodes():
    file_path = 'C:/Users/......./Resulats_sim/BC_Boundary/Nodes_Fix_nr.txt'
    return np.loadtxt(file_path, dtype=int)


def plot_positions(ax, positions, fixed_nodes, elements, highlight_node=None):
    ax.clear()  # Clear previous plot data
    global scatter
    scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=60, color='blue', marker='o', zorder=1, picker=True)

    if fixed_nodes is not None:
        ax.scatter(positions[fixed_nodes, 0], positions[fixed_nodes, 1], positions[fixed_nodes, 2], color='black', marker='x', s=150, zorder=2)

    if elem_on == 1:
        for elem in elements:
            vertices = positions[elem]
            for i in range(3):
                x_vals = [vertices[i][0], vertices[i + 1][0]]
                y_vals = [vertices[i][1], vertices[i + 1][1]]
                z_vals = [vertices[i][2], vertices[i + 1][2]]
                ax.plot(x_vals, y_vals, z_vals, color='cyan', linewidth=0.5)

            # Connect the last vertex to the first
            x_vals = [vertices[-1][0], vertices[0][0]]
            y_vals = [vertices[-1][1], vertices[0][1]]
            z_vals = [vertices[-1][2], vertices[0][2]]
            ax.plot(x_vals, y_vals, z_vals, color='cyan', linewidth=0.5)

    if highlight_node is not None:
        ax.scatter(positions[highlight_node, 0], positions[highlight_node, 1], positions[highlight_node, 2], color='red', marker='o', s=100, zorder=3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def on_pick(event):
    global selected_node
    if event.artist != scatter:
        return

    N = len(event.ind)
    if not N: return

    # only the first picked node is considered
    selected_node = event.ind[0]
    print(f"Selected node: {selected_node}")
    plot_positions(ax, positions, fixed_nodes, elements, highlight_node=selected_node)
    fig.canvas.draw()


def on_key(event):
    global selected_node, positions
    if selected_node is None or event.key.lower() != 'b':
        return

    # Update node position based on mouse coordinates
    x, y = event.xdata, event.ydata
    if x is not None and y is not None:
        # Adjust this transformation as needed for your data
        new_position = ax.transData.inverted().transform((x, y)) + positions[selected_node, 2]
        positions[selected_node, :2] = new_position[:2]

        plot_positions(ax, positions, fixed_nodes, elements, highlight_node=selected_node)
        fig.canvas.draw()


def main():
    global ax, fig, positions, fixed_nodes, elements
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    current_size = fig.get_size_inches()
    fig.set_size_inches(current_size * 2)

    ax.view_init(elev=40, azim=-110)

    fixed_nodes = load_fixed_nodes()
    positions = load_init_positions()
    elements = load_init_elements()

    plot_positions(ax, positions, fixed_nodes, elements)

    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()


if __name__ == '__main__':
    main()
