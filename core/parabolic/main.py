import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix, csr_matrix
from pydantic import BaseModel


# Setup logging
logging.basicConfig(level=logging.INFO)

class GeometricInfo(BaseModel):
    area: float
    edges: np.ndarray
    normals: np.ndarray
    barycenter: np.ndarray
    vertices: np.ndarray

    class Config:
        arbitrary_types_allowed = True


def generate_uniform_mesh(num_elements_per_edge: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Generate a uniform mesh of size num_elements_per_edge x num_elements_per_edge
    (Simple square domain)

    Parameters:
        num_elements_per_edge (int): Number of elements per edge

    Returns:
        vertices (np.array): Array of vertices
        elements (np.array): Array of elements
    """


    nx = ny = num_elements_per_edge
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    vertices = np.column_stack((X.flatten(), Y.flatten()))
    
    elements = []
    for i in range(ny-1):
        for j in range(nx-1):
            idx = i*nx + j
            elements.append([idx, idx+1, idx+nx])
            elements.append([idx+1, idx+nx+1, idx+nx])
    elements = np.array(elements)

    logging.info(f"Generated mesh with {len(vertices)} vertices and {len(elements)} elements")

    return vertices, elements

class Preprocessing:
    def __init__(self, vertices: np.ndarray, elements: np.ndarray, k: int = 1):
        self.vertices = vertices
        self.elements = elements
        self.k = k  # polynomial degree
        self.geom_info = {}  # store geometric information
        self.poly_basis = {}  # store polynomial basis info

    def compute_geometric_info(self):
        """ Compute geometric information for each element.
        Calculates the area, edges, normals, barycenter, and vertices for each element.

        Parameters:
            None

        Returns:
            None
        """
        for el_idx, element in enumerate(self.elements):
            el_vertices = self.vertices[element]
            
            # Compute element area
            area = self.polygon_area(el_vertices)
            
            # Compute element edges and normals
            edges, normals = self.compute_edges_and_normals(el_vertices)
            
            # Compute barycenter
            barycenter = np.mean(el_vertices, axis=0)

            self.geom_info[el_idx] = GeometricInfo(
                area=area,
                edges=edges,
                normals=normals,
                barycenter=barycenter,
                vertices=el_vertices
            )

    def define_polynomial_basis(self):
        """ Define the polynomial basis for P_k(K).
        For k=1, basis is {1, x, y}
        For k=2, basis is {1, x, y, x², xy, y²}
        For k=3, basis is {1, x, y, x², xy, y², x³, x²y, xy², y³}

        Parameters:
            None

        Returns:
            None
        """
        # For k=1, basis is {1, x, y}
        def basis_functions(x, y, xc, yc, h):
            """Return values of basis functions at (x,y).

            Parameters:
                x (float): x-coordinate
                y (float): y-coordinate
                xc (float): x-coordinate of the barycenter
                yc (float): y-coordinate of the barycenter
                h (float): element size

            Returns:
                basis_functions (np.array): Array of basis functions
            """
            return np.array([
                1.0,  # constant
                (x - xc)/h,  # scaled x
                (y - yc)/h   # scaled y
            ])
        
        def basis_gradients(x, y, xc, yc, h):
            """Return gradients of basis functions at (x,y)

            Parameters:
                x (float): x-coordinate
                y (float): y-coordinate
                xc (float): x-coordinate of the barycenter
                yc (float): y-coordinate of the barycenter
                h (float): element size

            Returns:
                basis_gradients (np.array): Array of basis gradients
            """
            return np.array([
                [0.0, 0.0],      # grad of constant
                [1.0/h, 0.0],    # grad of x
                [0.0, 1.0/h]     # grad of y
            ])

        self.poly_basis['functions'] = basis_functions
        self.poly_basis['gradients'] = basis_gradients
        self.poly_basis['size'] = 3  # number of basis functions for k=1

    @staticmethod
    def polygon_area(vertices)->float:
        """Compute polygon area using shoelace formula.
        
        Parameters:
            vertices (np.array): Array of vertices

        Returns:
            area (float): Area of the polygon
        """
        x = vertices[:,0]
        y = vertices[:,1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    @staticmethod
    def compute_edges_and_normals(vertices)->Tuple[np.ndarray, np.ndarray]:
        """Compute edges and their normal vectors
        
        Parameters:
            vertices (np.array): Array of vertices

        Returns:
            edges (np.array): Array of edges
            normals (np.array): Array of normals
        """
        n_vertices = len(vertices)
        edges = []
        normals = []
        for i in range(n_vertices):
            j = (i + 1) % n_vertices
            edge = vertices[j] - vertices[i]
            edges.append(edge)
            # Counter-clockwise normal
            normal = np.array([-edge[1], edge[0]])
            normal = normal / np.linalg.norm(normal)
            normals.append(normal)
        return np.array(edges), np.array(normals)
    
    def get_boundary_nodes(self)->np.ndarray:
        """Find boundary nodes where Dirichlet BCs will be applied.
        
        Parameters:
            None

        Returns:
            boundary_nodes (np.array): Array of boundary nodes
        """
        boundary_nodes = []
        for i, (x, y) in enumerate(self.vertices):
            if np.isclose(x, 0) or np.isclose(x, 1) or np.isclose(y, 0) or np.isclose(y, 1):
                boundary_nodes.append(i)
        return np.array(boundary_nodes)
    

class LocalMatrices:
    def __init__(self, preprocessing: Preprocessing, use_weighted_proj: bool = False):
        self.pre = preprocessing
        self.use_weighted_proj = use_weighted_proj

    def compute_polynomial_matrices(self, el_idx: int)->Tuple[np.ndarray, np.ndarray]:
        """Compute the polynomial matrices A^K and M^K.
        
        Parameters:
            el_idx (int): Index of the element

        Returns:
            A_poly (np.array): Polynomial stiffness matrix
            M_poly (np.array): Polynomial mass matrix
        """
        geom = self.pre.geom_info[el_idx]
        area = geom.area
        xc, yc = geom.barycenter
        h = np.sqrt(area)  # characteristic size
        
        # Retrieve basis functions and gradients
        n_basis = self.pre.poly_basis['size']
        basis_funcs = self.pre.poly_basis['functions']
        gradients = self.pre.poly_basis['gradients'](xc, yc, xc, yc, h)  # Evaluate at barycenter

        # Setup weighted projection
        if self.use_weighted_proj:
            w_k = 1/h
        else:
            w_k = 1
        
        # Compute polynomial mass matrix M^K with proper scaling
        M_poly = np.zeros((n_basis, n_basis))
        for i in range(n_basis):
            for j in range(n_basis):
                # Integral of basis function products over element
                M_poly[i, j] = w_k * self._integrate_basis_product(basis_funcs, i, j, xc, yc, h)
        
        # Compute polynomial stiffness matrix A^K with correct scaling
        A_poly = np.zeros((n_basis, n_basis))
        for i in range(1, n_basis):  # Skip the first row/column (constant function)
            for j in range(1, n_basis):
                A_poly[i, j] = area *np.dot(gradients[i], gradients[j])  # Correct scaling

        
        return A_poly, M_poly
    
    def compute_transformation_matrix(self, el_idx)->np.ndarray:
        """Compute the DOF-to-polynomial transformation matrix B
        
        Parameters:
            el_idx (int): Index of the element

        Returns:
            B (np.array): DOF-to-polynomial transformation matrix
        """
        geom = self.pre.geom_info[el_idx]
        vertices = geom.vertices
        area = geom.area
        xc, yc = geom.barycenter
        h = np.sqrt(area)
        
        # Compute B matrix with proper scaling
        n_vertices = len(vertices)
        n_basis = self.pre.poly_basis['size']
        B = np.zeros((n_basis, n_vertices))  # Note: transposed from before

        w_k = 1/h if self.use_weighted_proj else 1
        
        # Evaluate basis functions at vertices with proper scaling
        for i, vertex in enumerate(vertices):
            # Scale coordinates properly
            x_scaled = (vertex[0] - xc)
            y_scaled = (vertex[1] - yc)
            
            B[:,i] = [w_k*1.0,            # constant term
                     w_k*x_scaled,       # x term
                     w_k*y_scaled]       # y term
        
        return B
    
    def compute_stability_matrix(self, el_idx)->Tuple[np.ndarray, np.ndarray]:
        """Compute the stability matrix S^K.
        
        Parameters:
            el_idx (int): Index of the element

        Returns:
            S_A (np.array): Stabilization matrix for A
            S_M (np.array): Stabilization matrix for M
        """
        geom = self.pre.geom_info[el_idx]
        n_vertices = len(geom.vertices)
        area = geom.area
        
        # Use smaller stability parameter
        return (0.01 * area / n_vertices) * np.eye(n_vertices), (0.01 * area / n_vertices) * np.eye(n_vertices)
    
    def compute_element_matrices(self, el_idx, alpha: float = 0.01, beta: float = 0.01)->Tuple[np.ndarray, np.ndarray]:
        """Compute final local matrices.
        
        Parameters:
            el_idx (int): Index of the element
            alpha (float): Stiffness parameter
            beta (float): Mass parameter

        Returns:
            A (np.array): Stiffness matrix
            M (np.array): Mass matrix
        """
        A_poly, M_poly = self.compute_polynomial_matrices(el_idx)
    
        # Get transformation matrix
        B = self.compute_transformation_matrix(el_idx)
        
        # Get stability matrices (from paper)
        S_A, S_M = self.compute_stability_matrix(el_idx)
        
        # Compute consistency terms
        A_consistency = B.T @ A_poly @ B
        M_consistency = B.T @ M_poly @ B
        
        # Add stabilization terms (paper's formula)
        A = 0.00000001*A_consistency + alpha * S_A
        M = M_consistency + beta * S_M
        
        return A, M
    
    def _integrate_basis_product(self, basis_funcs, i, j, xc, yc, h):
        """Compute the integral of basis functions over the element.
        
        Parameters:
            basis_funcs (function): Basis functions
            i (int): Index of the first basis function
            j (int): Index of the second basis function
            xc (float): x-coordinate of the barycenter
            yc (float): y-coordinate of the barycenter
            h (float): element size

        Returns:
            integral (float): Integral of the basis functions
        """
        area = h * h  # For square elements
        if i == 0 and j == 0:  
            return area  # Scale by area
        elif i == 0 or j == 0:  
            return 0.0
        else:  
            return area/12 if i == j else 0.0  # Scale by area

    
            
class VEMParabolic:
    def __init__(self, vertices: np.ndarray, elements: np.ndarray, k: int = 1, use_weighted_proj: bool = False):
        # Initialize preprocessing
        self.pre = Preprocessing(vertices, elements, k)
        self.pre.compute_geometric_info()
        self.pre.define_polynomial_basis()

        # Setup weighted projection
        self.use_weighted_proj = use_weighted_proj
        
        # Initialize local matrices computer
        self.local_matrices = LocalMatrices(self.pre, self.use_weighted_proj)
        
        # Global matrices
        self.M = None
        self.A = None

    def assemble_global_matrices(self):
        """Assemble the global matrices M and A.
        
        Parameters:
            None

        Returns:
            None
        """
        n_vertices = len(self.pre.vertices)
        M = lil_matrix((n_vertices, n_vertices))
        A = lil_matrix((n_vertices, n_vertices))
        
        # Assemble matrices element by element
        for el_idx in range(len(self.pre.elements)):
            # Get element matrices
            A_local, M_local = self.local_matrices.compute_element_matrices(el_idx)
            
            # Get global indices
            dofs = self.pre.elements[el_idx]
            
            # Assemble using R_K^T A^K R_K
            for i, ii in enumerate(dofs):
                for j, jj in enumerate(dofs):
                    M[ii,jj] += M_local[i,j]
                    A[ii,jj] += A_local[i,j]
        
        self.M = M.tocsr()
        self.A = A.tocsr()

    def solve_backward_euler(self, T: float, nt: int, u0: np.ndarray, f: Callable[[float, float, float], float]) -> np.ndarray:
        """Time Discretization using Backward Euler with VEM-consistent source term.
        
        Parameters:
            T (float): Final time
            nt (int): Number of time steps
            u0 (np.array): Initial condition
            f (Callable[[float, float, float], float]): Source term

        Returns:
            U (np.array): Solution array
        """
        if self.M is None or self.A is None:
            self.assemble_global_matrices()
        
        dt = T/nt
        n = len(self.pre.vertices)
        U = np.zeros((nt+1, n))  # This creates array of correct size
        U[0] = u0
        
        # Precompute polynomial projection matrices for all elements
        projection_matrices = []
        for el_idx in range(len(self.pre.elements)):
            geom = self.pre.geom_info[el_idx]
            B = self.local_matrices.compute_transformation_matrix(el_idx)
            M_poly = self.local_matrices.compute_polynomial_matrices(el_idx)[1]
            projection_matrices.append((B, M_poly))
        
        # Set initial condition to zero on boundary
        boundary_nodes = self.pre.get_boundary_nodes()
        U[0, boundary_nodes] = 0.0
        
        # Precompute system matrix with boundary conditions
        system_matrix = self.M/dt + self.A
        for i in boundary_nodes:
            system_matrix[i, :] = 0.0
            system_matrix[:, i] = 0.0
            system_matrix[i, i] = 1.0
        
        # Time stepping
        for i in range(nt):
            t = (i+1)*dt
            b = (self.M @ U[i])/dt  # Previous time step contribution
            
            # Add source term using VEM projection
            for el_idx in range(len(self.pre.elements)):
                geom = self.pre.geom_info[el_idx]
                dofs = self.pre.elements[el_idx]
                area = geom.area
                B, M_poly = projection_matrices[el_idx]
                
                # Setup weighted projection
                if self.use_weighted_proj:
                    h = np.sqrt(area)
                    w_k = 1/h
                else:
                    w_k = 1

                # Project f onto P_k(K)
                f_proj_coeffs = np.zeros(self.pre.poly_basis['size'])
                for p_idx in range(self.pre.poly_basis['size']):
                    xc, yc = geom.barycenter
                    f_proj_coeffs[p_idx] = w_k * f(t, xc, yc) * area
                
                f_local = B.T @ np.linalg.solve(M_poly, f_proj_coeffs)
                b[dofs] += dt * f_local
            
            # Set RHS to zero at boundary nodes
            b[boundary_nodes] = 0.0
            
            # Solve system
            U[i+1] = spsolve(system_matrix, b)
            
            # Ensure boundary conditions are exactly satisfied
            U[i+1, boundary_nodes] = 0.0
        
        return U


    def solve_crank_nicolson(self, T: float, nt: int, u0: np.ndarray, f: Callable[[float, float, float], float]) -> np.ndarray:
        """Time Discretization using Crank-Nicolson.
        
        Parameters:
            T (float): Final time
            nt (int): Number of time steps
            u0 (np.array): Initial condition
            f (Callable[[float, float, float], float]): Source term

        Returns:
            U (np.array): Solution array
        """
        if self.M is None or self.A is None:
            self.assemble_global_matrices()
        
        dt = T / nt
        n = len(self.pre.vertices)
        U = np.zeros((nt + 1, n))
        U[0] = u0
        
        # Precompute projection matrices and system matrices
        projection_matrices = []
        for el_idx in range(len(self.pre.elements)):
            geom = self.pre.geom_info[el_idx]
            B = self.local_matrices.compute_transformation_matrix(el_idx)
            M_poly = self.local_matrices.compute_polynomial_matrices(el_idx)[1]
            projection_matrices.append((B, M_poly))
        
        # Precompute boundary nodes and system matrix
        boundary_nodes = self.pre.get_boundary_nodes()
        system_matrix = 0.5 * self.A + self.M / dt
        system_matrix = system_matrix.tolil()
        for i in boundary_nodes:
            system_matrix[i, :] = 0.0
            system_matrix[i, i] = 1.0
        system_matrix = system_matrix.tocsr()
        
        # Time stepping
        for i in range(nt):
            t_n = i * dt
            t_np1 = (i + 1) * dt
            
            # Compute source terms at t_n and t_{n+1/2}
            b = (self.M / dt - 0.5 * self.A) @ U[i]
            
            # Add source term contribution (midpoint rule)
            for el_idx in range(len(self.pre.elements)):
                geom = self.pre.geom_info[el_idx]
                dofs = self.pre.elements[el_idx]
                area = geom.area
                B, M_poly = projection_matrices[el_idx]

                # Setup weighted projection
                if self.use_weighted_proj:
                    h = np.sqrt(area)
                    w_k = 1/h
                else:
                    w_k = 1
                
                # Evaluate source at midpoint t_{n+1/2}
                t_mid = t_n + 0.5 * dt
                xc, yc = geom.barycenter
                f_mid = w_k * f(t_mid, xc, yc)

                # Project and integrate
                f_proj_coeffs = np.array([f_mid * area, 0.0, 0.0])  # For k=1
                f_local = B.T @ np.linalg.solve(M_poly, f_proj_coeffs)
                b[dofs] += dt * f_local  # Integrate over [t_n, t_{n+1}]
            
            # Apply boundary conditions to RHS
            b[boundary_nodes] = 0.0  # Enforce u=0 on boundary
            
            # Solve
            U[i + 1] = spsolve(system_matrix, b)
            U[i + 1, boundary_nodes] = 0.0  # Ensure exact BCs
        
        return U
    

# Add the error function
def compute_error(vem, u_h, u_exact)->float:
    """Compute ||u_h - u_exact||_M / ||u_exact||_M error.
    
    Parameters:
        vem (VEMParabolic): VEM solver
        u_h (np.array): Approximated solution
        u_exact (np.array): Exact solution

    Returns:
        error (float): Error
    """
    e = u_h - u_exact
    norm_e = np.sqrt(e @ vem.M @ e)
    norm_u = np.sqrt(u_exact @ vem.M @ u_exact)
    return norm_e / norm_u

# Modified run_solver()
def run_solver():
    # num_elements_per_edge = 16
    num_elements_per_edge = 4
    vertices, elements = generate_uniform_mesh(num_elements_per_edge)

    # Use weighted projection
    use_weighted_proj = True   

    # Initialize solver
    vem = VEMParabolic(vertices, elements, use_weighted_proj=use_weighted_proj)
    vem.assemble_global_matrices()  # Assemble M and A
    
    # Initial condition: u(0,x,y) = sin(πx)sin(πy)
    u0 = np.zeros(len(vertices))
    for i, (x, y) in enumerate(vertices):
        u0[i] = np.sin(np.pi*x) * np.sin(np.pi*y)
    
    # Source term: f(t,x,y) = e^t * sin(πx)sin(πy) * (1 + π²)
    def f(t, x, y):
        return np.exp(t) * np.sin(np.pi*x) * np.sin(np.pi*y) * (1 + np.pi**2)
    
    # Solve
    T = 1.0
    nt = 5
    # nt = 1655
    # U = vem.solve_backward_euler(T, nt, u0, f)
    U = vem.solve_crank_nicolson(T, nt, u0, f)
    
    # Compute exact solution for comparison
    u_exact = np.zeros(len(vertices))
    for i, (x, y) in enumerate(vertices):
        u_exact[i] = np.exp(T) * np.sin(np.pi*x) * np.sin(np.pi*y)
        
    # Compute error using the paper's discrete norm
    error = compute_error(vem, U[-1], u_exact)
    print(f"Paper's discrete L² error: {error:.4e}")
    
    return U, vertices, elements, u_exact

def plot_solution(U, vertices, elements, time_idx=-1):
    """Visualize the solution
    
    Parameters:
        U (np.array): Solution array
        vertices (np.array): Array of vertices
        elements (np.array): Array of elements
        time_idx (int): Index of the time step to plot

    Returns:
        None
    """
    import matplotlib.tri as tri
    
    # Create triangulation
    triang = tri.Triangulation(vertices[:,0], vertices[:,1], elements)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.tripcolor(triang, U[time_idx], shading='gouraud')
    plt.colorbar(label='u')
    plt.title(f'VEM Solution at t={1.0}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()

    

if __name__ == "__main__":
    U, vertices, elements, u_exact = run_solver()
    # plot_solution(U, vertices, elements)

    error = np.linalg.norm(U[-1] - u_exact) / np.linalg.norm(u_exact)
    print(f"Relative L2 error at  {error}")