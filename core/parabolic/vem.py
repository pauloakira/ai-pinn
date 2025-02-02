import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

class Preprocessing:
    """Step 1: Preprocessing (Mesh and Polynomial Basis)"""
    
    def __init__(self, vertices, elements, k=1):
        self.vertices = vertices
        self.elements = elements
        self.k = k  # polynomial degree
        self.geom_info = {}  # store geometric information
        self.poly_basis = {}  # store polynomial basis info
        
    def compute_geometric_info(self):
        """1.2 Compute Geometric Information"""
        for el_idx, element in enumerate(self.elements):
            el_vertices = self.vertices[element]
            
            # Compute element area
            area = self.polygon_area(el_vertices)
            
            # Compute element edges and normals
            edges, normals = self.compute_edges_and_normals(el_vertices)
            
            # Compute barycenter
            barycenter = np.mean(el_vertices, axis=0)
            
            # Store geometric info for this element
            self.geom_info[el_idx] = {
                'area': area,
                'edges': edges,
                'normals': normals,
                'barycenter': barycenter,
                'vertices': el_vertices
            }
    
    def define_polynomial_basis(self):
        """1.3 Define the Polynomial Basis for P_k(K)"""
        # For k=1, basis is {1, x, y}
        def basis_functions(x, y, xc, yc, h):
            """Return values of basis functions at (x,y)"""
            return np.array([
                1.0,  # constant
                (x - xc)/h,  # scaled x
                (y - yc)/h   # scaled y
            ])
        
        def basis_gradients(x, y, xc, yc, h):
            """Return gradients of basis functions at (x,y)"""
            return np.array([
                [0.0, 0.0],      # grad of constant
                [1.0/h, 0.0],    # grad of x
                [0.0, 1.0/h]     # grad of y
            ])
        
        self.poly_basis['functions'] = basis_functions
        self.poly_basis['gradients'] = basis_gradients
        self.poly_basis['size'] = 3  # number of basis functions for k=1
        
    @staticmethod
    def polygon_area(vertices):
        """Compute polygon area using shoelace formula"""
        x = vertices[:,0]
        y = vertices[:,1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    @staticmethod
    def compute_edges_and_normals(vertices):
        """Compute edges and their normal vectors"""
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

class LocalMatrices:
    """Step 2: Compute Local Element Matrices"""
    
    def __init__(self, preprocessing):
        self.pre = preprocessing
        
    def compute_polynomial_matrices(self, el_idx):
        """2.1.1 Compute the polynomial matrices A^K and M^K"""
        geom = self.pre.geom_info[el_idx]
        area = geom['area']
        xc, yc = geom['barycenter']
        h = np.sqrt(area)  # characteristic size
        
        n_basis = self.pre.poly_basis['size']
        
        # Compute polynomial mass matrix M^K with proper scaling
        M_poly = np.zeros((n_basis, n_basis))
        M_poly[0,0] = area
        M_poly[1:,1:] = area/12 * np.eye(2)
        
        # Compute polynomial stiffness matrix A^K with correct scaling
        A_poly = np.zeros((n_basis, n_basis))
        A_poly[1:,1:] = area * np.eye(2)  # Remove h^2 scaling
        
        return A_poly, M_poly
    
    def compute_transformation_matrix(self, el_idx):
        """2.1.1 Compute the DOF-to-polynomial transformation matrix B"""
        geom = self.pre.geom_info[el_idx]
        vertices = geom['vertices']
        area = geom['area']
        xc, yc = geom['barycenter']
        h = np.sqrt(area)
        
        # Compute B matrix with proper scaling
        n_vertices = len(vertices)
        n_basis = self.pre.poly_basis['size']
        B = np.zeros((n_basis, n_vertices))  # Note: transposed from before
        
        # Evaluate basis functions at vertices with proper scaling
        for i, vertex in enumerate(vertices):
            # Scale coordinates properly
            x_scaled = (vertex[0] - xc)
            y_scaled = (vertex[1] - yc)
            
            B[:,i] = [1.0,            # constant term
                     x_scaled/h,       # x term
                     y_scaled/h]       # y term
        
        return B
    
    def compute_stability_matrix(self, el_idx):
        """2.1.2 Compute the stability matrix S^K"""
        geom = self.pre.geom_info[el_idx]
        n_vertices = len(geom['vertices'])
        area = geom['area']
        
        # Use smaller stability parameter
        return (0.01 * area / n_vertices) * np.eye(n_vertices)
    
    def compute_element_matrices(self, el_idx):
        """2.1.3 Compute final local matrices"""
        # Get polynomial matrices
        A_poly, M_poly = self.compute_polynomial_matrices(el_idx)
        
        # Get transformation matrix
        B = self.compute_transformation_matrix(el_idx)
        
        # Get stability matrix
        S = self.compute_stability_matrix(el_idx)
        
        # Compute consistency terms with proper matrix multiplication order
        A_consistency = B.T @ A_poly @ B
        M_consistency = B.T @ M_poly @ B
        
        # Add stability terms with smaller parameters
        alpha = 0.01  # reduced from 0.1
        beta = 0.01   # reduced from 0.1
        A = A_consistency + alpha * S
        M = M_consistency + beta * S
        
        return A, M

class VEMParabolic:
    """Main VEM solver class"""
    
    def __init__(self, vertices, elements):
        """Initialize VEM solver"""
        # Initialize preprocessing
        self.pre = Preprocessing(vertices, elements)
        self.pre.compute_geometric_info()
        self.pre.define_polynomial_basis()
        
        # Initialize local matrices computer
        self.local_matrices = LocalMatrices(self.pre)
        
        # Global matrices
        self.M = None
        self.A = None
    
    def assemble_global_matrices(self):
        """Step 3: Assemble Global Matrices"""
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
    
    def solve_backward_euler(self, T, nt, u0, f):
        """Step 4: Time Discretization using Backward Euler with proper scaling"""
        if self.M is None or self.A is None:
            self.assemble_global_matrices()
        
        dt = T/nt
        n = len(self.pre.vertices)
        U = np.zeros((nt+1, n))
        U[0] = u0
        
        # Time stepping with proper scaling of matrices
        for i in range(nt):
            t = (i+1)*dt
            
            # Compute right-hand side
            b = (self.M @ U[i])/dt
            
            # Add source term with proper integration
            for j in range(n):
                x, y = self.pre.vertices[j]
                b[j] += dt * f(t, x, y)  # Note: added dt factor
            
            # Solve system
            system_matrix = self.M/dt + self.A
            U[i+1] = spsolve(system_matrix, b)
        
        return U

def test_vem_solver():
    """Test the VEM implementation with exact solution"""
    # Create mesh (simple square domain)
    nx = ny = 32
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
    
    # Initialize solver
    vem = VEMParabolic(vertices, elements)
    
    # Initial condition: u(0,x,y) = sin(πx)sin(πy)
    u0 = np.zeros(len(vertices))
    for i, (x, y) in enumerate(vertices):
        u0[i] = np.sin(np.pi*x) * np.sin(np.pi*y)
    
    # Source term: f(t,x,y) = e^t * sin(πx)sin(πy) * (1 + π²)
    def f(t, x, y):
        return np.exp(t) * np.sin(np.pi*x) * np.sin(np.pi*y) * (1 + np.pi**2)
    
    # Solve
    T = 1.0
    nt = 1000
    U = vem.solve_backward_euler(T, nt, u0, f)
    
    # Compute exact solution for comparison
    u_exact = np.zeros(len(vertices))
    for i, (x, y) in enumerate(vertices):
        u_exact[i] = np.exp(T) * np.sin(np.pi*x) * np.sin(np.pi*y)
        
    # Compute error
    error = np.linalg.norm(U[-1] - u_exact) / np.linalg.norm(u_exact)
    print(f"Relative L2 error at t={T}: {error}")
    
    return U, vertices, elements, u_exact

def plot_solution(U, vertices, elements, time_idx=-1):
    """Visualize the solution"""
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
    # Run test
    U, vertices, elements, u_exact = test_vem_solver()
    
    # Plot solution
    plot_solution(U, vertices, elements)
    
    # Print some info
    print("Solution shape:", U.shape)