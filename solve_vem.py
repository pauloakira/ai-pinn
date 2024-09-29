import numpy as np
import core.vem as vem

def solve_1d(nodes, elements, supp, E, A, I, load, q, t):
    # loads
    # load = np.array([[2,3],[3,4]])
    # q = -400
    # t = 0
    f_dist = vem.buildBeamDistributedLoad(load,t,q,nodes)

    # stiffness matrix
    K = vem.buildGlobaBeamK(nodes, elements, E, A, I, 1)

    # apply DBC
    K, f = vem.applyDBCBeam(K, f_dist, supp)

    # solve
    print()
    print("######################### Beam ##########################")
    uh_vem = np.linalg.solve(K,f)
    print(uh_vem)
    print("#########################################################")

    return uh_vem, K, f