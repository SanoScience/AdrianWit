import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import TensorDataset, DataLoader
from Create_batches import Create_batches


#######################################################################################################
# Model definitions by PyTorch
#######################################################################################################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(6, 128)   # 6 wejść (3 pozycje + 3 przemieszczenia), 128 neuronów w warstwie
        self.fc2 = nn.Linear(128, 64)  # 64 neurony w drugiej warstwie
        self.fc3 = nn.Linear(64, 3)    # 3 wyjścia (3 pozycje)

    def forward(self, x):
        x = f.relu(self.fc1(x))  # ReLU
        x = f.relu(self.fc2(x))  # ReLU
        x = self.fc3(x)
        return x


net = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

#######################################################################################################
# Read batches
#######################################################################################################

print('Read batches:')
config_file_path = 'C:/Users/Sharkoon/Documents/SOFA_Python_Project/SOFA_gene_config.txt'
with open(config_file_path, 'r') as file:
    lines = file.readlines()

scenarios_nr = int(lines[0].split('=')[1].strip())
iterations_nr = int(lines[1].split('=')[1].strip())
path_to_position_files = 'C:/Users/Sharkoon/Documents/SOFA_Python_Project/Resulats_sim/position/'
path_to_forces_files = 'C:/Users/Sharkoon/Documents/SOFA_Python_Project/Resulats_sim/forces/'

batch_num = scenarios_nr * iterations_nr
print(batch_num)
batches = Create_batches(scenarios_nr, iterations_nr, path_to_position_files, path_to_forces_files)

# Wczytanie danych z batches
all_input_data = np.vstack([batch[0] for batch in batches])
all_output_data = np.vstack([batch[1] for batch in batches])

# Podział na zestawy treningowe i testowe
test_size = 0.1  # 10% danych na zestaw testowy
split_index = int(all_input_data.shape[0] * (1 - test_size))

X, y = all_input_data[:split_index], all_output_data[:split_index]
X_test, y_test = all_input_data[split_index:], all_output_data[split_index:]

# Konwersja danych na tensory PyTorch
X_tensor = torch.Tensor(X)
y_tensor = torch.Tensor(y)
X_test_tensor = torch.Tensor(X_test)
y_test_tensor = torch.Tensor(y_test)

# Tworzenie TensorDataset
train_dataset = TensorDataset(X_tensor, y_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Parametry DataLoadera
batch_size = 545  # constant for liver model

# Tworzenie DataLoaderów
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#######################################################################################################
# Network Trainig
#######################################################################################################
print('Start Training')

num_epochs = 5
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):

        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Zapisywanie modelu
torch.save(net.state_dict(), 'network_PyTorch.pkl')

#######################################################################################################
# Prediction
#######################################################################################################

# Wczytywanie model:
net.load_state_dict(torch.load('network_PyTorch.pkl'))
# Ustawienie modelu w tryb ewaluacji
net.eval()


# Display the "i" batch from the test dataset
i = 11
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        if i:
            print("Inputs:", inputs)
            print("Ground:", labels)
            predicted_outputs = net(inputs)
            print("Prediction:", predicted_outputs)
            break  # Exit the loop after processing the batch


'''
# Jeśli chcesz przewidzieć wartości dla konkretnej próbki danych, upewnij się, że jest ona w odpowiednim formacie tensora
input_data_pred = []
predykcja = torch.Tensor(input_data_pred)
wynik_predykcji = net(predykcja)
print(wynik_predykcji)
'''