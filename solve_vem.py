import time
import numpy as np
import core.vem as vem
import utils.mesh as mesh

def solve_1d(nodes, elements, supp, E, A, I, load, q, t, verbose=True):
    f_dist = vem.buildBeamDistributedLoad(load,t,q,nodes)

    # stiffness matrix
    K = vem.buildGlobaBeamK(nodes, elements, E, A, I, 1)

    # apply DBC
    K, f = vem.applyDBCBeam(K, f_dist, supp)

    # Start timing the solution
    start = time.time()

    # Solve the linear system
    uh_vem = np.linalg.solve(K,f)

    # End timing the solution
    end = time.time()

    solving_time = end - start

    if verbose:
        print(f"Solving time [s]: {solving_time:.4f}")  
        print("######################### Beam ##########################")
        print(uh_vem)
        print("#########################################################")

    return uh_vem, K, f, solving_time

if __name__ == "__main__":
    # Define the number of elements per edge
    num_elements_per_edge = 64

    # geometry data
    L = 2.0
    I = 1e-4
    A = 1

    # material data
    E = 27e6

    # Define load parameters
    q = -400
    t = 0

    # Generate the geometry
    nodes, elements, supp, load = mesh.generate_portic_geometry(num_elements_per_edge, L)

    # Define the material parameters
    E = 210e9
    A = 0.01
    I = 0.0001

    # Solve the problem using the VEM
    uh_vem, K, f, solving_time = solve_1d(nodes, elements, supp, E, A, I, load, q, t)

    print(f"Solving time [s]: {solving_time:.4f}")
    print("######################### Beam ##########################")
    print(uh_vem)
    print("#########################################################")