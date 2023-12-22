import numpy as np
import os
import torch

###################################################################################################################################
#     Load learnig data (4, 50, 3, 545, x)
###################################################################################################################################

# Read config file
scenarios_nr = []
iter_nr = []

# Otwarcie pliku i odczytanie zawartości
with open('C:/Users/Sharkoon/Documents/SOFA_Python_Project/SOFA_gene_config.txt', 'r') as file:
    for line in file:
        key, value = line.split('=')
        key = key.strip()
        value = value.strip()

        if key == 'scenarios_nr':
            scenarios_nr = int(value)
        elif key == 'iter_nr':
            iter_nr = int(value)

####################################################

data_position_path = f"C:/Users/Sharkoon/Documents/SOFA_Python_Project/Resulats_sim/position/"
all_data = []

# Iteracja przez każdy scenariusz (s) i punkt czasu (t)
for s in range(scenarios_nr):
    scenario_data = []
    for t in range(iter_nr):
        file_path = f"{s}_{t}.txt"
        full_path = os.path.join(data_position_path, file_path)

        # Ładowanie danych pozycji (545, 3)
        position_data = np.loadtxt(full_path)
        # Transpozycja danych pozycji (3, 545)
        position_data = np.transpose(position_data)
        # Rozszerzenie wektora pozycji do macierzy (3, 545, 545)
        position_data_2 = np.repeat(position_data[:, :, np.newaxis], axis=2)

        scenario_data.append(position_data_2)

    # Konwersja listy scenario_data na czterowymiarową tablicę numpy (50, 3, 545, x)
    scenario_array = np.stack(scenario_data, axis=0)
    all_data.append(scenario_array)

# Konwersja listy all_data na pięciowymiarową tablicę numpy (4, 50, 3, 545, x)
all_data_array = np.stack(all_data, axis=0)

print(all_data_array.shape)
print('(scenariusz, iteracja, kanał xyz, węzeł, węzeł)')

# Konwersja danych na tensory PyTorch:
all_data_tensor = torch.tensor(all_data_array, dtype=torch.float32)

# Zapisz tensor do pliku (format .pt)
#tensor_file_path = 'C:/Users/Sharkoon/Documents/SOFA_Python_Project/Resulats_sim/all_data_tensor.pt'
#torch.save(all_data_tensor, tensor_file_path)
