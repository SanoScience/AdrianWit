import numpy as np


# Function to load data from a file
def load_data(file_path):
    return np.loadtxt(file_path)


def Create_batches(scenarios_nr, iterations_nr, path_to_position_files, path_to_forces_files):
    batches = []  # Inicjalizacja listy batchy
    for s in range(0, scenarios_nr):
        # Load force data
        force_file = f'{path_to_forces_files}f_{s}.txt'
        forces = load_data(force_file)
        # Read the first value from the forces file
        first_force_value = int(forces[0])

        for n in range(0, iterations_nr):
            # Load position data
            position_file = f'{path_to_position_files}{s}_{n}.txt'

            positions = load_data(position_file)
            input_liver_pos = positions[:, :3]
            output_liver_pos = positions[:, 3:]

            difference = np.round(np.subtract(output_liver_pos, input_liver_pos), 8)

            # Create a modified difference array where all rows are zero except the one indicated by the first force value
            difference_zero = np.zeros_like(difference)
            difference_zero[first_force_value] = difference[first_force_value]

            # Create a batch
            input_batch = np.concatenate((input_liver_pos, difference_zero), axis=1)
            output_batch = output_liver_pos  # Ground truth

            batches.append((input_batch, output_batch))

    return batches
