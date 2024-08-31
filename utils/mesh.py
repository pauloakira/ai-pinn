import numpy as np

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