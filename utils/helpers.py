import re
import json
import numpy as np
from math import sqrt, pow, atan2

def calcArea(coords):
    ''' Calculate area using Shoelace formula.

    Input:
        - nodes(np.array): coordinates of element nodes.

    Output:
        - area(float): polygon area. 
    '''
    area = 0

    for i in range(len(coords)-1):
        x1 = coords[i][0] - coords[0][0]
        x2 = coords[i+1][0] - coords[0][0]
        y1 = coords[i][1] - coords[0][1]
        y2 = coords[i+1][1] - coords[0][1]
        area += + x1 * y2 - x2 * y1
    
    area /= 2
    #print(f'Area = {area}')
    return abs(area)

def calcCentroid(coords):
    ''' Calculate the centroid of a polygon.

    Input:
        - coords(np.array): coordinates of element nodes.

    Output:
        - c(np.array): centroid of a polygon.
    '''
    coord_exapanded = np.concatenate((coords, [coords[0]]), axis = 0)
    x = coord_exapanded [:,0]
    y = coord_exapanded [:,1]
    cx = 0
    cy = 0
    area = calcArea(coords)
    for i in range(len(coords)):
        cx += (x[i] + x[i+1])*(x[i]*y[i+1] - x[i+1]*y[i])
        cy += (y[i] + y[i+1])*(x[i]*y[i+1] - x[i+1]*y[i])
    cx = 1/(6*area)*cx
    cy = 1/(6*area)*cy
        
    c = [cx, cy]
    return c

def calcLength(coord):
    ''' Calculate the length of an edge.

    Input:
        - coord(np.array): coordinates of nodes that consitutes an 
        edge.

    Output:
        -length(float): length of the edge.    
    '''
    
    x = coord[1][0] - coord[0][0]
    y = coord[1][1] - coord[0][1]
    length = sqrt(pow(coord[1][0]-coord[0][0],2)+pow(coord[1][1]-coord[0][1],2))

    return length
    
def calcPolygonalDiam(m, coords):
    ''' Calculate the polygonal diameter of a single element. The polygonal 
    diameter is the largest distance between any pair of vertices.

    Input:
        - m(int): number of vertices on an element.
        - coords(np.array): coordinates of element nodes.
    
    Output:
        - h(float): polygonal diameter of the element.
    '''
    h = 0
    for i in range(m):
        for j in range(m):
            x2 = pow(coords[i][0]-coords[j][0],2)
            y2 = pow(coords[i][1]-coords[j][1],2)
            d = sqrt(x2+y2)
            if d > h:
                h = d
    #print(f'h = {h}')
    return h


def computeNormalVector(coord):
    ''' Calculate the outward normal vector of a line, using the 
    formulae:
    
        n = (y, -x),
    
    where y = y2 - y1 and x = x2 - x1.

    Input:
        - coord(np.array): coordinates of nodes that consitutes an 
        edge.
    
    Output:
        - n(np.array): normal vectors related to each edge of an
        element.
    '''
    for i in range(len(coord)-1):
        x = coord[i+1][0] - coord[i][0]
        y = coord[i+1][1] - coord[i][1]
        normal = np.array([y, -x])
        n = np.linalg.norm(normal)
        #print(f'Normal = {normal/n}')
    return normal/n

def scaledCoords(coord, c, h):
    ''' Calculate the scaled coordinates, such that:
    1- the origin of the coordinate aces is the centroid of the
    element;
    2- all the coordinates values are scaled by the polygonal 
    diameter h.

    Input:
        - coord(np.array): coordinates of the node.
        - c(np.array): centroid of the element.
        - h(float): polygonal diameter of the element.

    Output:
        - xi, eta(float): scaled coordinates.
    '''
    xi = (coord[0]-c[0])/h
    eta = (coord[1]-c[1])/h

    return xi, eta

def getScalarIndices(nodesInd):
    n = len(nodesInd)
    dofs = np.zeros(n, dtype=int)
    dofs[0:n:1] = nodesInd
    return dofs

def getDofsIndices(nodesInd):
    ''' Given a list with the node indices, it returns the global indexation
    regarding the correspondent degree of freedom.

    Input:
        - nodesIns(np.array): vector with node indices.

    Output:
        - dofs(np.array): list of degrees of freedom global indexation.
    '''

    n = 2*len(nodesInd)
    dofs = np.zeros(n, dtype=int)
    dofs[0:n:2] = 2*nodesInd
    dofs[1:n:2] = 2*nodesInd + 1

    return dofs

def calcAngle(coord):
    ''' Calculate the inclination angle of an element.

    Input:
        - coord(np.array): coordinates of nodes that constitues an
        edge

    Output:
        - theta(float): inclination angle of the element.
    '''

    x = coord[1][0] - coord[0][0]
    y = coord[1][1] - coord[0][1]
    theta = atan2(y,x)

    return theta

def getOrder2Indices(nodeInd, momentInd):
    ''' Given a list with the node indices, it returns the global indexation
    regarding the correspondent degree of freedom.

    Input:
        - nodesIns(np.array): vector with node indices from a single
        element.
        - momentInd(int): index for the moment component (ne-control).

    Output:
        - dofs(np.array): list of degrees of freedom global indexation.
    '''
    n = 2*len(nodeInd) + 1
    dofs = np.zeros(n, dtype=int)
    dofs[0] = 2*nodeInd[0]
    dofs[1] = 2*nodeInd[0] + 1
    dofs[2] = 2*nodeInd[1]
    dofs[3] = 2*nodeInd[1] + 1

    # moment 
    dofs[4] = momentInd

    return dofs

def getOrder1IndicesBeam(nodeInd):
    ''' Given a list with the node indices, it returns the global indexation
    regarding the correspondent degree of freedom.

    Input:
        - nodesIns(np.array): vector with node indices.

    Output:
        - dofs(np.array): list of degrees of freedom global indexation.
    '''
    n = 3*len(nodeInd)
    dofs = np.zeros(n, dtype=int)
    dofs[0:n:3] = 3*nodeInd
    dofs[1:n:3] = 3*nodeInd + 1
    dofs[2:n:3] = 3*nodeInd + 2 

    return dofs

def getOrder2IndicesBeam(nodeInd, momentInd):
    ''' Given a list with the node indices, it returns the global indexation
    regarding the correspondent degree of freedom. This is used for the
    second order Euler-Bernoulli beam.

    Input:
        - nodesIns(np.array): vector with node indices from a single
        element.
        - momentInd(int): index for the moment component (ne-control).

    Output:
        - dofs(np.array): list of degrees of freedom global indexation.
    '''
    n = 3*len(nodeInd) + 2
    dofs = np.zeros(n, dtype=int)
    dofs[0] = 3*nodeInd[0]
    dofs[1] = 3*nodeInd[0] + 1
    dofs[2] = 3*nodeInd[0] + 2
    dofs[3] = 3*nodeInd[1]
    dofs[4] = 3*nodeInd[1] + 1
    dofs[5] = 3*nodeInd[1] + 2

    # moment
    dofs[6] = momentInd
    dofs[7] = momentInd + 1

    return dofs

def suppArrayConstruction(supp1, supp2):
    ''' Concatanate the support lists into a single array. The 
    supp1 list refers to horizontal restrictions and the supp2
    list refers to the vertical restrictions.

    Input:
        - supp1(list): horizontal constraints.
        - supp2(list): vertical constraints.
    
    Output:
        - supp(np.array): array with the restricted nodes.
    '''
    for i,s1 in enumerate(supp1):
        if i == 0:
            supp = np.array([[s1, 1, 0]])
        else:
            supp = np.concatenate((supp, np.array([[s1, 1, 0]])), axis = 0)
    
    for j, s2 in enumerate(supp2):
        if len(supp) == 0 and j == 0:
            supp = np.array([[s2, 0, 1]])
        else:
            supp = np.concatenate((supp, np.array([[s2, 0, 1]])), axis = 0)
    
    return supp


def readGeometry(filename):
    ''' Given a text file with nodes (coordinates), elements, support restrictions
    and distributed load segment, this function is responsible to return the formated
    input using numpy arrays. 

    Input:
        - filename(string): path + filename containing the geometry specifications.

    Output:
        - elements(np.array): contains the element indexation.
        - nodes(np.array): contains the coordinates of the nodes.
        - supp(np.array): array with the restricted nodes.
        - load(np.array): array containing the edges indices.
    '''
    f = open(filename,"r")
    lines = f.readlines()

    # skip the "Nodes" and "x    y" lines
    i = 2 
    
    # pattern for remove \n
    pattern = r'\n'

    # read nodes
    while lines[i] != '\n':
        coord = re.sub(pattern, '', lines[i])
        coord = re.split(' +',coord)
        x = float(coord[0])
        y = float(coord[1])
        if i == 2:
            nodes = np.array([[x, y]])
        else:
            nodes = np.concatenate((nodes, np.array([[x, y]])), axis = 0)
        i += 1

    # read elements
    if lines[i] != 'Elements':        
        i += 2 # go to the first line containing the indexation
    
    first_iteration = True # control if it is the first iteration

    while lines[i] != '\n':
        el = re.sub(pattern,'', lines[i])
        el = re.split(' +', el)
        el = [int(e) - 1 for e in el]
        el = np.array(el)
        if first_iteration:
            elements = np.array([el])
            first_iteration = False
        else:
            elements = np.concatenate((elements, [el]),axis = 0)
        i += 1

    # read the first support set (Supp1)
    if lines[i] != 'Supp1':
        i += 2

    supp1 = []

    while lines[i] != '\n':
        s1 = re.sub(pattern, '', lines[i])
        s1 = re.split(' +', s1)
        s1 = int(s1[0]) - 1
        supp1.append(s1)
        i += 1

    # read the second support set (Supp2)
    if lines[i] != 'Supp2':
        i += 2
    
    supp2 = []

    while lines[i] != '\n':
        s2 = re.sub(pattern, '', lines[i])
        s2 = re.split(' +', s2)
        s2 = int(s2[0]) - 1
        supp2.append(s2)
        i += 1

    # build the supp array
    supp = suppArrayConstruction(supp1, supp2)

    # read the distributed load indexes
    if lines[i] != 'Load':
        i += 2
    
    first_iteration = True

    while i < len(lines) and lines[i] != '\n' :
        l = re.sub(pattern, '', lines[i])
        l = re.split(' +', l)        
        startIndex = int(l[0]) - 1
        endIndex = int(l[1]) - 1
        if first_iteration:
            load = np.array([[startIndex, endIndex]])
            first_iteration = False
        else:
            load = np.concatenate((load, np.array([[startIndex, endIndex]])), axis = 0)
        i += 1

    f.close()

    return nodes, elements, supp, load

def readParameter(filename):
    ''' Read json file containing the material, load and order parameters.

    Input:
        - filename(string): name of the file with the respective extension.

    Output:
        - E(float): elastic modulus.
        - nu(float): Poisson coefficient.
        - I(float): inertia moment.
        - A(area): area of the transversal section.
        - qx(float): distributed load in x axis.
        - qy(float): distributed load in y axis.
        - order(int): formulation order.
    '''
    # open the json file
    f = open(filename)

    # return the json data as a dictionary
    parameter = json.load(f)

    # close the json file
    f.close()

    E = parameter['E']
    nu = parameter['nu']
    I = parameter['I']
    A = parameter['A']
    
    qx = parameter['qx']
    qy = parameter['qy']

    k = parameter['k']

    return E, nu, I, A, qx, qy, k

def transform2Vectors(u, model_type, order):
    ''' Transform the concataneted solution vector into the correspondent 
    individual solution vectors.

    Model types:
        - 2dVem [Order(s): 1] -> Elastic 2D implementatation under the plane
        state hypothesis,
    
    Input:
        - u(np.array): concataneted solution vector.
        - model_type(string): VEM model type.
        - order(int): model order.

    Output:
        - individual solution vectors.
    '''
    if model_type == "2dVem":
        if order == 1:
            n = len(u)
            uh = u[0:n:2]
            vh = u[1:n:2]
            return uh, vh
        else:
            raise("Order not implemented.")
    else:
        raise("Model not implemented!")

def saveOutputTxt(filepath, uh, vh, model_type, order, theta=None):
    ''' Save the displacement field into text files.

    Model types:
        - 2dVem [Order(s): 1] -> Elastic 2D implementatation under the plane
        state hypothesis,
    
    Input:
        - filepath(string): path + <name>.txt
        - u(np.array): concataneted solution vector.
        - model_type(string): VEM model type.
        - order(int): model order.

    Output:
        None.
    '''
    if model_type == "2dVem":
        if order == 1:
            data = np.column_stack([uh, vh])
            np.savetxt(filepath, data, fmt=['%f', '%f'])
        else:
            raise("Order not implemented.")
    else:
        raise("Model not implemented!")

def nextIterationVector(vec):
    ''' Build the next iteration vector.

    Example: 
        vec = [A, B, C, D]
        v1 = [D, A, B, C]
    
    Input:
        - vec(np.array); original vector.
    
    Output:
        -v1(np.array): next iteration vector
    '''
    v1 = np.zeros(len(vec))
    for i in range(len(vec)):
        if i == 0:
            v1[i] = vec[-1]
        else:
            v1[i] = vec[i-1]
    return v1

def horizontalBarDisc(bar_length, num_elements):
    ''' Discretize a horizontal bar/beam/cable by receiving the length of 
    the geometry and the number of desired elements. It is assumed only
    the one-dimensional case. And the mesh is uniform.

    Input:
        - bar_length(float): length of the bar.
        - num_elements(int): number of elements in the final mesh.

    Output:
        - nodes(np.array): array with the nodes coordinates.
        - elements(np.array): array with the elements nodes indices.
    '''
    
    # initialize nodes array with zeros
    nodes = np.zeros(shape=(num_elements+1,2))

    # initialize element array with zeros
    elements = np.zeros(shape=(num_elements,2), dtype=int)

    # mesh increment
    inc = float(bar_length/num_elements)
    coord = inc

    # initialize index controler
    i = 1

    while coord <= bar_length:
        nodes[i,0] = coord
        i+=1
        coord += inc
    
    for i in range(num_elements):
        elements[i,0] = i
        elements[i,1] = i + 1

    return nodes, elements