import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

class VEMParabolic:
    def __init__(self, mesh, k=1):
        """
        Initialize VEM solver for parabolic problems
        
        Parameters:
        mesh: Dictionary containing mesh information
            vertices: nx2 array of vertex coordinates
            elements: mxk array of vertex indices for each element
        k: polynomial degree (default=1)
        """
        self.mesh = mesh
        self.k = k
        self.vertices = mesh['vertices']
        self.elements = mesh['elements']
        self.n_vertices = len(mesh['vertices'])
        self.n_elements = len(mesh['elements'])
        
        # Initialize matrices
        self.M = None  # Mass matrix
        self.S = None  # Stiffness matrix
        
    def compute_element_matrices(self, element_vertices):
        """Compute element mass and stiffness matrices"""
        n_vertices = len(element_vertices)
        area = self.compute_polygon_area(element_vertices)
        
        # Mass matrix
        ME = np.zeros((n_vertices, n_vertices))
        # For k=1, use consistent mass matrix
        for i in range(n_vertices):
            for j in range(n_vertices):
                if i == j:
                    ME[i,j] = area / 6
                else:
                    ME[i,j] = area / 12
        
        # Stiffness matrix
        SE = np.zeros((n_vertices, n_vertices))
        # Compute gradients for k=1
        for i in range(n_vertices):
            for j in range(n_vertices):
                if i == j:
                    # Get neighbors
                    prev = element_vertices[(i-1)%n_vertices]
                    next = element_vertices[(i+1)%n_vertices]
                    curr = element_vertices[i]
                    
                    # Compute gradients
                    dx_prev = curr[0] - prev[0]
                    dy_prev = curr[1] - prev[1]
                    dx_next = next[0] - curr[0]
                    dy_next = next[1] - curr[1]
                    
                    SE[i,j] = (dx_prev*dx_prev + dy_prev*dy_prev + 
                              dx_next*dx_next + dy_next*dy_next) / (2*area)
                else:
                    # Check if vertices are neighbors
                    if abs(i-j) == 1 or abs(i-j) == n_vertices-1:
                        vi = element_vertices[i]
                        vj = element_vertices[j]
                        dx = vj[0] - vi[0]
                        dy = vj[1] - vi[1]
                        SE[i,j] = -(dx*dx + dy*dy) / (2*area)
        
        return ME, SE
    
    def compute_polygon_area(self, vertices):
        """Compute area of polygon using shoelace formula"""
        x = vertices[:,0]
        y = vertices[:,1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    def assemble_global_matrices(self):
        """Assemble global mass and stiffness matrices"""
        n = self.n_vertices
        M = lil_matrix((n, n))
        S = lil_matrix((n, n))
        
        for el in range(self.n_elements):
            # Get element vertices
            vertex_indices = self.elements[el]
            element_vertices = self.vertices[vertex_indices]
            
            # Compute element matrices
            ME, SE = self.compute_element_matrices(element_vertices)
            
            # Add to global matrices
            for i in range(len(vertex_indices)):
                for j in range(len(vertex_indices)):
                    ii = vertex_indices[i]
                    jj = vertex_indices[j]
                    M[ii,jj] += ME[i,j]
                    S[ii,jj] += SE[i,j]
        
        self.M = M.tocsr()
        self.S = S.tocsr()
    
    def solve_backward_euler(self, T, nt, u0, f):
        """
        Solve parabolic problem using backward Euler method
        
        Parameters:
        T: Final time
        nt: Number of time steps
        u0: Initial condition
        f: Source term function f(t,x,y)
        
        Returns:
        U: Solution at all time steps
        """
        if self.M is None or self.S is None:
            self.assemble_global_matrices()
            
        dt = T/nt
        n = self.n_vertices
        U = np.zeros((nt+1, n))
        U[0] = u0
        
        # Time stepping
        for i in range(nt):
            t = (i+1)*dt
            
            # Compute load vector
            b = self.M.dot(U[i])
            for j in range(n):
                x, y = self.vertices[j]
                b[j] += dt * f(t, x, y)
            
            # Solve linear system (M + dt*S)u^{n+1} = b
            A = self.M + dt*self.S
            U[i+1] = spsolve(A, b)
            
        return U

def test_vem_solver():
    """Test the VEM solver with a known solution"""
    # Create a simple rectangular mesh
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    X, Y = np.meshgrid(x, y)
    vertices = np.column_stack((X.flatten(), Y.flatten()))
    
    # Create triangular elements
    elements = []
    nx, ny = 4, 4
    for i in range(ny):
        for j in range(nx):
            # Lower triangle
            elements.append([i*(nx+1) + j, 
                          i*(nx+1) + j + 1, 
                          (i+1)*(nx+1) + j])
            # Upper triangle
            elements.append([i*(nx+1) + j + 1, 
                          (i+1)*(nx+1) + j + 1, 
                          (i+1)*(nx+1) + j])
    
    mesh = {
        'vertices': vertices,
        'elements': np.array(elements)
    }
    
    # Initialize solver
    vem = VEMParabolic(mesh)
    
    # Set initial condition
    u0 = np.zeros(len(vertices))
    for i, (x, y) in enumerate(vertices):
        u0[i] = np.sin(np.pi*x) * np.sin(np.pi*y)
    
    # Define source term
    def f(t, x, y):
        return np.exp(t) * np.sin(np.pi*x) * np.sin(np.pi*y)
    
    # Solve
    T = 1.0
    nt = 100
    U = vem.solve_backward_euler(T, nt, u0, f)
    
    return U

if __name__ == "__main__":
    U = test_vem_solver()
    print("Solution shape:", U.shape)
    print(U)