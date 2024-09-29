import time
import numpy as np
import core.vem as vem

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