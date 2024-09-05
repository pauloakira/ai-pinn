# Python libs
import numpy as np

# Custom libs
import utils.helpers as utils

##################### POISSON EQUATION #########################
def buildLocalPoisson(coords, use_stabilization = True):
    ''' Build the local stiffness matrix for Poisson Equation for
    the linear case.

    Input:
        - coords(np.array): coordinates of the element.

    Output:
        - Kloc(np.array): local stiffness matrix.
    '''

    # number of vertices for each element
    m = len(coords)

    # auxiliary matrices
    B = np.zeros(shape=(3, m))
    D = np.zeros(shape=(m, 3))
    I = np.identity(m)

    # polygonal diameter 
    h = utils.calcPolygonalDiam(m, coords)

    # gradients of the scaled monomial
    grad2 = np.array([1/h, 0])
    grad3 = np.array([0, 1/h])

    # normal vector
    en = np.zeros(2)

    # calculate the centroid
    c = utils.calcCentroid(coords)

    # extended coordinates vector
    coord_exapanded = np.concatenate((coords, [coords[0]]), axis = 0)
    x = coord_exapanded[:,0]
    y = coord_exapanded[:,1]
    for i in range(m):

        # normalization
        if i == 0:
            en[0] = y[i+1] - y[m-1]
            en[1] = -x[i+1] + x[m-1]
        else:
            en[0] = y[i+1] - y[i-1]
            en[1] = -x[i+1] + x[i-1]

        # B matrix
        B[0,i] = 1/m
        B[1,i] = 0.5 * en[0] * grad2[0]
        B[2,i] = 0.5 * en[1] * grad3[1]
        
        # D matrix
        D[i,0] = 1
        D[i,1] = (coords[i,0]-c[0])/h
        D[i,2] = (coords[i,1]-c[1])/h

    # compute G and modified G (denoted by GG)
    G = B @ D
    GG = B @ D
    GG[0,:] = 0
    G_inv = np.linalg.inv(G)

    # computing projectors
    proj1 = G_inv @ B
    proj2 = D @ proj1

    # compute the consistency part
    Kc = np.transpose(proj1) @ GG @ proj1 


    if not use_stabilization:
        return Kc
    
    # compute the stability part
    Ks = np.transpose(I-proj2)@(I-proj2)

    # assembly local matrix
    Kloc = Kc + Ks

    return Kloc

def buildGlobalPoisson(nodes, elements, use_stabilization = True):
    ''' Build the global stiffness matrix for Poisson Equation for
    the linear case.

    Input:
        - elements(np.array): contains the element indexation.
        - nodes(np.array): contains the coordinates of the nodes.

    Output:
        - K(np.array): global stiffness matrix.
    '''
    ndof = len(nodes)
    K = np.zeros(shape=(ndof, ndof))
    for e in elements:
        e_dofs = utils.getScalarIndices(e)
        coords = nodes[e,:]
        Kloc = buildLocalPoisson(coords, use_stabilization)
        K[np.ix_(e_dofs, e_dofs)] += Kloc
    return K

def computePoissonLoad(f, coords):
    ''' Computhe the load value in each node by using the centroid
    of the element.

    Input:
        - f(function): load function.
        - coords(np.array): coordinates of the element.

    Output:
        - load(float): load value in a node.
    '''
    # number of vertices
    m = len(coords)

    # calculate area
    area = utils.calcArea(coords)

    # calculate the centroid
    c = utils.calcCentroid(coords)

    load = area/m * f(c)

    return load

def buildPoissonLoad(f, nodes, elements):
    ''' Build the global load vecotr.

    Input:
        - f(function): load function.
        - elements(np.array): contains the element indexation.
        - nodes(np.array): contains the coordinates of the nodes.
    
    Output:
        - F(np.array): global load vector
    '''
    ndof = len(nodes)

    F = np.zeros(ndof)

    for e in elements:
        e_dofs = utils.getScalarIndices(e)
        coords = nodes[e,:]
        F[np.ix_(e_dofs)] += computePoissonLoad(f,coords)

    return F

def applyPoissonDBC(K, F, supp):
    ''' Apply Dirichlet boundary conditions in the specified nodes.

    Input:
        - K(np.aray): global stiffness matrix.
        - F(np.array): global load vector.
        - supp(np.array): array with the restricted nodes.
    
    Output:
        - K(np.array): new stiffness matrix.
        - F(np.array): new load vector.
    '''
    for s in supp:
        K[:,s] = 0
        K[s,:] = 0
        K[s,s] = 1
        F[s] = 0
    return K, F

##################### 1D EULER-BERNOULLI #########################

def buildLocalBeamK(coord, E, A, I, order):
    ''' Build beam local stiffeness matrix.

    Input:
        - coord(np.array): coordinates of the nodes from an element.
        - E(float): elastic modulus.
        - A(area): area of the transversal section.
        - I(floar): inertia moment.
        - order(int): formulation order.

    Output:
        - Kloc(np.array): local matrix.
    '''

    l = utils.calcLength(coord)
    theta = utils.calcAngle(coord)
    c = np.cos(theta)
    s = np.sin(theta)

    print(f"Coords: {coord} - Length: {l}")

    if order == 1:

        # rotation matrix
        Q = np.zeros(shape=(6,6))
        Q[0][0] = c
        Q[0][1] = s
        Q[1][0] = -s
        Q[1][1] = c
        Q[2][2] = 1
        Q[3][3] = c
        Q[3][4] = s
        Q[4][3] = -s
        Q[4][4] = c
        Q[5][5] = 1

        # stiffness matrix
        K = np.zeros(shape=(6,6))
        K[0][0] = E*A/l
        K[0][3] = -E*A/l
        K[1][1] = 12*E*I/(l**3)
        K[1][2] = 6*E*I/(l**2)
        K[1][4] = -12*E*I/(l**3)
        K[1][5] = 6*E*I/(l**2)
        K[2][1] = 6*E*I/(l**2)
        K[2][2] = 4*E*I/l 
        K[2][4] = -6*E*I/l 
        K[2][5] = 2*E*I/l 
        K[3][0] = -E*A/l 
        K[3][3] = E*A/l 
        K[4][1] = -12*E*I/(l**3)
        K[4][2] = -6*E*I/(l**2)
        K[4][4] = 12*E*I/(l**3)
        K[4][5] = -6*E*I/(l**2)
        K[5][1] = 6*E*I/(l**2)
        K[5][2] = 2*E*I/l 
        K[5][4] = -6*E*I/(l**2)
        K[5][5] = 4*E*I/l 

        K = np.transpose(Q) @ K @ Q

        return K


    if order == 2:

        # rotation matrix
        Q = np.zeros(shape=(8,8))
        Q[0][0] = c
        Q[0][1] = -s
        Q[1][0] = s
        Q[1][1] = c
        Q[2][2] = 1
        Q[3][3] = c
        Q[3][4] = -s
        Q[4][3] = s
        Q[4][4] = c
        Q[5][5] = 1
        Q[6][6] = 1
        Q[7][7] = 1

        # stiffness matrix
        K = np.zeros(shape=(8,8))
        K[0][0] = 4*E*A/l
        K[0][3] = 2*E*A/l
        K[0][6] = -6*E*A/l

        K[1][1] = 192*E*I/(l**3)
        K[1][2] = 36*E*I/(l**3)
        K[1][4] = 168*E*I/(l**2)
        K[1][5] = -24*E*I/(l**2)
        K[1][7] = -360*E*I/(l**3)

        K[2][1] = K[1][2]
        K[2][2] = 9*E*I/(l**2)
        K[2][4] = 24*E*I/(l**2)
        K[2][5] = -3*E*I/l
        K[2][7] = -60*E*I/(l**2)
        
        K[3][0] = K[0][3]
        K[3][3] = 4*E*A/l
        K[3][6] = -6*E*A/l

        K[4][1] = K[1][4]
        K[4][2] = K[2][4]
        K[4][4] = 192*E*I/(l**3)
        K[4][5] = -36*E*I/(l**2)
        K[4][7] = -360*E*I/(l**3)

        K[5][1] = K[1][5]
        K[5][2] = K[2][5]
        K[5][4] = K[4][5]
        K[5][5] = 9*E*I/l
        K[5][7] = 60*E*I/(l**2)

        K[6][0] = K[0][6]
        K[6][3] = K[3][6]
        K[6][6] = 12*E*A/l

        K[7][1] = K[1][7]
        K[7][2] = K[2][7]
        K[7][4] = K[4][7]
        K[7][5] = K[5][7]
        K[7][7] = 720*E*I/(l**3)

        K = np.transpose(Q) @ K @ Q

        #print(K)

        return K
    else:
        raise("Not implemented for this order.")

def buildGlobaBeamK(nodes, elements, E, A, I, order):
    ''' Build the 1D beam global stiffness matrix.

    Input:
        - nodes(np.array): array with the nodes coordinates.
        - elements(np.array): array with the elements nodes indices.
        - E(float): elastic modulus.
        - A(area): area of the transversal section.
        - I(float): inertia moment.
        - order(int): formulation order.
    
    Output:
        - K(np.array): global tangencial stiffness matrix.
    '''
    if order == 1:
        ndof = 3 * len(nodes)
        K = np.zeros(shape=(ndof, ndof))
        for e in elements:
            e_dofs = utils.getOrder1IndicesBeam(e)
            coord = nodes[e, :]
            Kloc = buildLocalBeamK(coord, E, A, I, order)
            K[np.ix_(e_dofs, e_dofs)] += Kloc
        return K
    if order == 2:
        ne = len(elements)
        ndof = 3*len(nodes) + 2*ne
        K = np.zeros(shape=(ndof, ndof))
        momentInd = 3*len(nodes)
        for e in elements:
            e_dofs = utils.getOrder2IndicesBeam(e, momentInd)
            coord = nodes[e, :]
            Kloc = buildLocalBeamK(coord, E, A, I, order)
            K[np.ix_(e_dofs, e_dofs)] += Kloc
            momentInd = + 2
        
        return K
    else:
        raise("Not implemented for this order.")

def applyDBCBeam(K, f_dist, supp):
    ''' Apply Dirichlet boundary conditions in the specified nodes.

    Input:
        - K(np.aray): global stiffness matrix.
        - f_dist(np.array): distributed load vector.
        - supp(np.array): array with the restricted nodes.
    
    Output:
        - K(np.array): new stiffness matrix.
    '''
    for node in supp:
        if node[1] == 1:
            i = 3*node[0]
            K[:,i] = 0
            K[i,:] = 0
            K[i][i] = 1
            f_dist[i] = 0
        if node[2] == 1:
            j = 3*node[0]+1
            K[:,j] = 0
            K[j,:] = 0
            K[j][j] = 1
            f_dist[j] = 0
        if node[3] == 1:
            k = 3*node[0]+2
            K[:,k] = 0
            K[k,:] = 0
            K[k][k] = 1
            f_dist[k] = 0
    
    return K, f_dist

def buildBeamDistributedLoad(load, t, q, nodes):
    ''' Apply the Neumann boundary conditions (especifically the
    distributed load).

    Input:
        - load(np.array): array containing the edges indices.
        - qx(float): distributed load in x axis.
        - qy(float): distributed load in y axis.
        - nodes(np.array): array of coordinates regarding each node.

    Output:
        - f(np.array): global external load vector.
    '''
    ndof = 3*len(nodes)
    f = np.zeros(shape=(ndof))
    for e in load:
        coord = nodes[e,:]
        theta = utils.calcAngle(coord)
        c = np.cos(theta)
        s = np.sin(theta)

        # rotation matrix
        Q = np.zeros(shape=(6,6))
        Q[0][0] = c
        Q[0][1] = s
        Q[1][0] = -s
        Q[1][1] = c
        Q[2][2] = 1
        Q[3][3] = c
        Q[3][4] = s
        Q[4][3] = -s
        Q[4][4] = c
        Q[5][5] = 1
        l = utils.calcLength(coord)
        floc = np.array([t*l/2, q*l/2, q*l**2/12, t*l/2, q*l/2, -q*l**2/12])
        dofs = utils.getOrder1IndicesBeam(e)
        f[np.ix_(dofs)] += floc

    return f   

def applyPointwiseLoad(nodeInd, nodes, fx, fy, m):
    ''' Apply the pointwise load in a specific node. 

    Input:
        - nodeInd(int): node index.
        - nodes(np.array): array with the nodes coordinates.
        - fx(float): horizontal component of the load.
        - fy(float): vertical component of the load.
        - m(float): applied momentum.

    Output:
        - f(np.array): pointwise load vector.
    '''
    ndof = 3*len(nodes)
    f = np.zeros(ndof)
    f[3*nodeInd] = fx
    f[3*nodeInd + 1] = fy
    f[3*nodeInd + 2] = m

    return f