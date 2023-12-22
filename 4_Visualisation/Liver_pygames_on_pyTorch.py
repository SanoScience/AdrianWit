import pygame
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as f

#######################################################################################################
# Definicja modelu sieci
#######################################################################################################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Definicja warstw sieci
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        return self.fc3(x)

# Wczytywanie modelu
net = Net()
net.load_state_dict(torch.load('C:/Users/..../network_PyTorch.pkl'))
net.eval()

#######################################################################################################
# Funkcje pomocnicze gry
#######################################################################################################

def load_init_positions():
    file_path = 'C:/Users/........./Mesh/liver2.npz'
    liver2_node = np.load(file_path)
    return liver2_node['node']

def load_init_elements():
    file_path = 'C:/Users/........./Mesh/liver2.npz'
    liver2_elem = np.load(file_path)
    return liver2_elem['elem']

def load_fixed_nodes():
    file_path = 'C:/Users/......./Resulats_sim/BC_Boundary/Nodes_Fix_nr.txt'
    return np.loadtxt(file_path, dtype=int)

def draw_elements(screen, positions, elements, zoom_factor, offset, screen_height):
    for element in elements:
        for i in range(len(element)):
            start_pos = adjust_coordinates(positions[element[i-1]], zoom_factor, offset, screen_height)
            end_pos = adjust_coordinates(positions[element[i]], zoom_factor, offset, screen_height)
            pygame.draw.line(screen, (0, 255, 255), start_pos[:2], end_pos[:2], 1)

def adjust_coordinates(pos, zoom_factor, offset, screen_height):
    """Adjusts the coordinates for rendering, flipping the y-axis."""
    adjusted_x = (pos[0] * zoom_factor) + offset[0]
    adjusted_y = screen_height - ((pos[1] * zoom_factor) + offset[1])
    return np.array([adjusted_x, adjusted_y])

def calculate_displacement_vector(original_pos, current_pos):
    return [current_pos[0] - original_pos[0], current_pos[1] - original_pos[1], current_pos[2] - original_pos[2]]

def find_closest_node(positions, mouse_pos, threshold, zoom_factor, offset, screen_height):
    for i, pos in enumerate(positions):
        scaled_pos = adjust_coordinates(pos, zoom_factor, offset, screen_height)
        if math.hypot(scaled_pos[0] - mouse_pos[0], scaled_pos[1] - mouse_pos[1]) < threshold:
            return i
    return None

def update_positions(net, positions, displacement_vector, selected_node):
    # Tworzenie macierzy przesunięć
    displacements = np.zeros_like(positions)
    if selected_node is not None:
        # Przypisanie wektora przemieszczenia dla wybranego węzła
        displacements[selected_node, :] = displacement_vector

    # Połączenie pozycji z przesunięciami
    input_matrix = np.hstack((positions, displacements))
    print(input_matrix[selected_node, :])

    # Konwersja do tensora PyTorch
    input_tensor = torch.FloatTensor(input_matrix)

    # Przetwarzanie przez sieć neuronową
    with torch.no_grad():
        new_positions = net(input_tensor).numpy()

    return new_positions

#######################################################################################################
# Główna pętla gry
#######################################################################################################

pygame.init()
display_width = 1600
display_height = 900
screen = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption("Liver PyGames Visualization")
clock = pygame.time.Clock()

pygame.font.init()
font = pygame.font.SysFont(None, 30)

positions = load_init_positions()
original_positions = np.copy(positions)
elements = load_init_elements()
fixed_nodes = load_fixed_nodes()

zoom_factor = 450  # Constant zoom factor
screen_offset = np.array([600, -5200, 0])  # Adjusted screen offset for y-axis inversion
mouse_z = 0.0  # Constant Z value for mouse interaction

selected_node = None
last_mouse_pos = None

running = True
while running:
    fps = clock.get_fps()
    mouse_pos = pygame.mouse.get_pos()
    frame_displacement_vector = [0, 0, 0]

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos_3d = np.array([mouse_pos[0], mouse_pos[1], mouse_z])
            selected_node = find_closest_node(positions, mouse_pos_3d, 15, zoom_factor, screen_offset, display_height)
            last_mouse_pos = mouse_pos_3d
        elif event.type == pygame.MOUSEBUTTONUP:
            selected_node = None
        elif event.type == pygame.MOUSEMOTION:
            if selected_node is not None and selected_node not in fixed_nodes:
                scaled_mouse_x = (mouse_pos[0] - screen_offset[0]) / zoom_factor
                scaled_mouse_y = (display_height - mouse_pos[1] - screen_offset[1]) / zoom_factor
                positions[selected_node][0], positions[selected_node][1] = scaled_mouse_x, scaled_mouse_y
                if last_mouse_pos is not None:
                    frame_displacement_vector = [(mouse_pos[0] - last_mouse_pos[0]) / zoom_factor,
                                                 (mouse_pos[1] - last_mouse_pos[1]) / zoom_factor,
                                                 0]
                    if frame_displacement_vector == [0, 0, 0]:
                        frame_displacement_vector = [0.000001, 0.000002, 0.000000 ]
                last_mouse_pos = np.array([mouse_pos[0], mouse_pos[1], mouse_z])

    if selected_node is not None:
        # Aktualizacja pozycji węzłów za pomocą sieci neuronowej
        print(frame_displacement_vector)
        positions = update_positions(net, positions, frame_displacement_vector, selected_node)

    screen.fill((0, 0, 0))
    draw_elements(screen, positions, elements, zoom_factor, screen_offset, display_height)

    for i, pos in enumerate(positions):
        adjusted_pos = adjust_coordinates(pos, zoom_factor, screen_offset, display_height)
        color = (0, 255, 0) if i not in fixed_nodes else (255, 0, 0)
        if i == selected_node:
            color = (0, 0, 255)
        pygame.draw.circle(screen, color, adjusted_pos[:2], 5)

    # Wyswietlanie informacji o FPS i wybranym węźle
    fps_text = font.render(f"FPS: {fps:.2f}", True, (255, 255, 255))
    screen.blit(fps_text, (10, 10))

    if selected_node is not None:
        node_info_text = font.render(f"Node: {selected_node}", True, (255, 255, 255))
        screen.blit(node_info_text, (10, 40))

        # Use the original 3D position for displaying the initial position
        initial_3d_pos = original_positions[selected_node]
        initial_position_text = font.render(f"Initial Pos: [{initial_3d_pos[0]:.3f}, {initial_3d_pos[1]:.3f}, {initial_3d_pos[2]:.3f}]", True, (255, 255, 255))
        screen.blit(initial_position_text, (10, 70))

        displacement_vector = calculate_displacement_vector(original_positions[selected_node], positions[selected_node])
        displacement_text = font.render(f"Displacement: [{displacement_vector[0]:.4f}, {displacement_vector[1]:.4f}, {displacement_vector[2]:.4f}]", True, (255, 255, 255))
        screen.blit(displacement_text, (10, 100))

        frame_displacement_text = font.render(f"Frame Displacement: [{frame_displacement_vector[0]:.4f}, {frame_displacement_vector[1]:.4f}, {frame_displacement_vector[2]:.4f}]", True, (255, 255, 255))
        screen.blit(frame_displacement_text, (10, 130))

    pygame.display.update()
    clock.tick(60)

pygame.quit()






























