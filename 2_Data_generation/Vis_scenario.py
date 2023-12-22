import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import glob

# Zmienna do wyboru numeru testu (teraz - indeksu grawitacji)
index_choice = 9
save_gif = 0

# Zmienna do wyboru zakresu kolorów
color_range = (0, 0.23)  # zakres kolorów
color_map = 'rainbow'  # mapa kolorów

def load_fixed_nodes():
    file_path = 'C:/Users/Sharkoon/Documents/SOFA_Python_Project/Resulats_sim/BC_Boundary/Nodes_Fix_nr.txt'
    if os.path.exists(file_path):
        return np.loadtxt(file_path, dtype=int)
    else:
        print(f'File {file_path} does not exist.')
        return None

def load_positions(gravity_index, iteration):
    file_path = f'C:/Users/Sharkoon/Documents/SOFA_Python_Project/Resulats_sim/position/{gravity_index}_{iteration}.txt'
    if os.path.exists(file_path):
        return np.loadtxt(file_path, delimiter='\t')
    else:
        print(f'File {file_path} does not exist.')
        return None

def plot_positions(ax, positions, displacements, fixed_nodes, title, colorbar, color_range, color_map, save_gif):
    ax.clear()  # Clear previous plot data
    scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=displacements, cmap=color_map, marker='o', vmin=color_range[0], vmax=color_range[1], zorder=1)
    if fixed_nodes is not None:
        ax.scatter(positions[fixed_nodes, 0], positions[fixed_nodes, 1], positions[fixed_nodes, 2], color='black', marker='x', s=150, zorder=2)
    ax.set_title(title)
    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if not colorbar:
        plt.colorbar(scatter, ax=ax, label=' Displacement [-] ')

    if save_gif != 1:
        plt.draw()
        plt.pause(0.05)  # pause to allow the plot to update

def get_number_of_iterations(gravity_index):
    file_pattern = f'C:/Users/Sharkoon/Documents/SOFA_Python_Project/Resulats_sim/position/{gravity_index}_*.txt'
    files = glob.glob(file_pattern)
    return len(files)

def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Powiększanie okna o 2x
    current_size = fig.get_size_inches()
    fig.set_size_inches(current_size * 2)

    # Ustawienie początkowych wartości azymutu i elewacji
    ax.view_init(elev=40, azim=-110)

    fixed_nodes = load_fixed_nodes()  # Wczytaj numery węzłów z pliku tekstowego

    number_of_iterations = get_number_of_iterations(index_choice)
    initial_positions = load_positions(index_choice, 0)  # load initial positions
    all_displacements = []

    for iteration in range(number_of_iterations):
        positions = load_positions(index_choice, iteration)
        if positions is not None:
            displacements = np.linalg.norm(positions - initial_positions, axis=1)  # calculate displacement
            all_displacements.append(displacements)

    colorbar = False  # flag to add colorbar only once

    for iteration in range(number_of_iterations):
        positions = load_positions(index_choice, iteration)
        if positions is not None:
            title = f'Gravity Index: {index_choice}, Iteration: {iteration}'
            plot_positions(ax, positions, all_displacements[iteration], fixed_nodes, title, colorbar, color_range, color_map, save_gif)
            colorbar = True  # set flag to true after first iteration

    def update(iteration):
        positions = load_positions(index_choice, iteration)
        title = f'Gravity Index: {index_choice}, Iteration: {iteration}'
        if positions is not None:
            plot_positions(ax, positions, all_displacements[iteration], fixed_nodes, title, colorbar, color_range, color_map, save_gif)

    if save_gif == 1:
        # Zapisanie animacji jako plik wideo
        ani = animation.FuncAnimation(fig, update, frames=number_of_iterations, repeat=False)
        ani.save('C:/Users/Sharkoon/Videos/Sofa_position.gif', writer='pillow', fps=15)
    else:
        plt.show()  # Keep the plot open at the end


if __name__ == '__main__':
    main()
