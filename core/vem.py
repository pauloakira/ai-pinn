# Python libs
import numpy as np

# Custom libs
import utils.helpers as utils

##################### POISSON EQUATION #########################
def buildLocalPoisson(coords):
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

    # compute the stability part
    Ks = np.transpose(I-proj2)@(I-proj2)

    # assembly local matrix
    Kloc = Kc + Ks

    return Kloc

def buildGlobalPoisson(nodes, elements):
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
        Kloc = buildLocalPoisson(coords)
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