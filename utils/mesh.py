import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from dataclasses import dataclass
from scipy.spatial import Voronoi, voronoi_plot_2d

def unitary_square_mesh(self, h = 32):
    dh = 1/h

    N_LINE = int(1/dh + 1)
    N_GAP = int(1/dh)
    N_VERTEX = 4
    N_ELEM = N_GAP * N_GAP

    interval = np.linspace(0, 1, h+1)
    nodes = np.zeros(shape=(N_LINE*N_LINE, 2))
    k = 0
    for i in interval:
        for j in interval:
            nodes[k,0] = j
            nodes[k,1] = i
            k += 1

    elements = np.zeros(shape=(N_ELEM, 4), dtype=int)
    INIT = 0
    FINISH = N_LINE - 1
    v = 0
    MULTI = 2
    for j in range(0, N_GAP):
        for i in range(INIT, FINISH, 1):
            elements[v, 0] = i
            elements[v, 1] = i + 1
            elements[v, 2] = i + N_LINE + 1
            elements[v, 3] = i + N_LINE 
            v += 1
        INIT = INIT + N_LINE
        FINISH = MULTI * N_LINE - 1
        MULTI = MULTI + 1
    
    return nodes, elements

def generate_portic_geometry(num_elements_per_edge, L):
    """
    Generate a 2D portic geometry defining nodes, elements and supports.

    Parameters:
    num_elements_per_edge (int): Number of elements per edge.
    L (float): length of the each edge.

    Returns:
    nodes (np.ndarray): Node coordinates (n_nodes x 2)
    elements (np.ndarray): Element connectivity (n_elements x n_nodes_per_element)  
    supp (np.ndarray): Support definition (n_supports x 4) with columns [node_id, x_disp, y_disp, rot]
    load (np.ndarray): Load definition (n_loads x 2) with columns [element_id, load_value]
    """
    
    x_coords = np.linspace(0, L, num_elements_per_edge + 1)
    y_coords = np.linspace(0, L, num_elements_per_edge + 1)

    top_nodes = np.array([[x, L] for x in x_coords if x != 0])
    left_nodes = np.array([[0, y] for y in y_coords])
    right_nodes = np.array([[L, y] for y in reversed(y_coords) if y != L])

    nodes = np.vstack([left_nodes, top_nodes, right_nodes])

    elements = np.array([[i, i+1] for i in range(len(nodes)-1)])
    flatten_elements = elements.flatten()

    supp = np.array([[flatten_elements[0], 1, 1, 1], [flatten_elements[-1], 1, 1, 1]])

    # Find the edges (element connections) along the top
    load = []
    top_node_indices = [i for i, node in enumerate(nodes) if node[1] == L]  # Indices of top nodes

    for i, elem in enumerate(elements):
        if elem[0] in top_node_indices and elem[1] in top_node_indices:
            load.append(elem)  
    return nodes, elements, supp, load

def plot_nodes(nodes, elements):
    """
    Plot the nodes and elements of a 2D mesh.

    Parameters:
    nodes (np.ndarray): Node coordinates (n_nodes x 2)
    elements (np.ndarray): Element connectivity (n_elements x n_nodes_per_element)
    """

    _, ax = plt.subplots()
    
    # Plot nodes
    ax.plot(nodes[:, 0], nodes[:, 1], 'ro', label='Nodes')

    # Plot elements as lines connecting nodes
    for element in elements:
        element_coords = nodes[element]
        ax.plot(element_coords[:, 0], element_coords[:, 1], 'b-')

    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Custom Geometry Plot')
    plt.legend()
    plt.show()

def save_geometry_to_json(nodes, elements, supp, load, q0x, q0y, filename):
    """
    Save the geometry defined by nodes and elements to a JSON file.

    Parameters:
    nodes (np.ndarray): Node coordinates (n_nodes x 2)
    elements (np.ndarray): Element connectivity (n_elements x n_nodes_per_element)
    filename (str): Name of the JSON file to save
    """

    @dataclass
    class Supp:
        node: int
        uBound: float
        vBound: float
        rotationBound: float
        xCoord: float
        yCood: float

    supp_list = []
    for s in supp:
        supp_list.append(Supp(int(s[0]), float(s[1]), float(s[2]), float(s[3]), float(nodes[s[0]][0]), float(nodes[s[0]][1])).__dict__)

    @dataclass
    class Load:
        element: List[int]
        q0x: float
        q0y: float

    load_list = []
    for l in load:
        load_list.append(Load(l.tolist(), q0x, q0y).__dict__)

    data = {
        'nodes': nodes.tolist(),
        'elements': elements.tolist(),
        'dbc': supp_list,
        'nbc': load_list,
    }

    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Geometry data saved to {filename}")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

if __name__ == "__main__":
    # Define the number of elements per edge
    num_elements_per_edge = 32

    # geometry data
    L = 2.0
    I = 1e-4
    A = 1

    q0x = -400.0
    q0y = 0.0

    # Generate the geometry
    nodes, elements, supp, load = generate_portic_geometry(num_elements_per_edge, L)

    # Save the geometry to a JSON file
    save_geometry_to_json(nodes, elements, supp, load, q0x, q0y, f'data/geometries/portic_{num_elements_per_edge}.json')