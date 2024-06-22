# Physical-Informed Neural Networks (PINNs)

The Physical-Informed Neural Networks integrate the physical laws described by PDEs directly into the neural network training process. This is done by incorporating the PDEs into the loss function, which guides the network to learn solutions that respect these laws. A key feature of PINNs is the use of automatic differentiation to compute the derivatives of the neural network output with respect to its inputs. This is crucial for ensuring that the network’s predictions satisfy the PDEs.

The loss function in PINNs consists of two main component:
- **Data misfit**: This term measures the difference between the neural network’s predictions and the available training data.
- **PDE Residuals**: This term enforces the physical laws by measuring the residuals of the PDEs when the network’s predictions are substituted into them.

The combined loss function guides the training process, ensuring that the learned solutions are consistent with both the observed data and the physical laws. The network is trained using gradient-based optimization methods (e.g., stochastic gradient descent, Adam, L-BFGS). The gradients of the loss function with respect to the network parameters are computed using automatic differentiation.

Regarding data, the Physical-Informed Neural Networks, can be of two types:

- **Data-Driven Solutions**: PINNs can be used to find solutions to PDEs when the form of the PDE is known, and the goal is to find the specific solution that fits the data. Example: Solving the Burgers' equation to find the velocity field given initial and boundary conditions.

- **Data-Driven Discovery**: PINNs can also be used to discover the form of the PDE itself from data. This involves learning the parameters of the PDE that best describe the observed data. Example: Discovering the parameters of the Navier-Stokes equations from flow data.

## Data-drive solutions

### Continuous time models

We want to compute solutions of PDEs of the form 
$$
\begin{equation}
    u_t =\mathcal{N}[u] = 0,
\end{equation}
$$
$x\in \Omega$, $t\in [0,T]$, where $u(t,x)$ denotes the latent (hidden) solution, $\mathcal{N}[\cdot]$ is a nonlinear differential operator, and $\Omega \subset \mathbb{R}^d$. The latent (hidden) solution is the true underlying function or state of the system that we aim to recover from limited and noisy observations. The term "hidden" emphasizes that we do not have direct access to the complete function over its entire domain. Instead, we have discrete and possibly noisy observations. The task is to reconstruct the continuous function from these observations.

Let's define
$$

\begin{equation}
    f(t,x)=u_t + \mathcal{N}[u],
\end{equation}
$$
and proceed by approximating $u(t,x)$ by a deep neural network. This setup results in a physics informed neural network $f(x,t)$. The idea to allocate both the 
initial and boundary conditions, and to enforce the physical constraints dictated by the partial differential equation, is to divide the loss function $\mathcal{L}$ in two parts: 
$$
\begin{equation}
    \mathcal{L} = \mathcal{L}_u + \mathcal{L}_f,
\end{equation} 
$$
where $\mathcal{L}_u$ measures how well the neural network's predictions match the observed data (it is defined over the training data points, which typically include initial and boundary conditions) and $\mathcal{L}_f$ enforces the PDE constraints by measuring the residuals of the PDE at a set of collocation points (he residuals are computed by substituting the neural network's predictions into the PDE). The combined loss ensures that the neural network fits the observed data well and satisfies the physical laws described by the differential equation simuntaneously. It is important to mention that the PDE residuals are evaluated at a set of collocation points, which are selected randomly or using a space-filling design, but these points are not fixed like a grid in classical methods.

Accordingly to the authors:

*...unlike any classical numerical method for solving partial dfferential equations, this prediction is obtained without any sort of discretization of the spatio-temporal domain.*

The neural network provides a smooth and continuous approximation of the solution over the entire domain. Instead of approximating derivatives using differences, PINNs use automatic differentiation to compute the exact derivatives of the neural network’s output with respect to its inputs. Discretization methods face challenges in higher dimensions due to the curse of dimensionality (exponential growth of computational cost with the number of dimensions) but PINNs can handle higher-dimensional problems more gracefully, as the network’s complexity does not scale exponentially with the dimension.

### Example 1 - The Burger's Equation

In one space dimension, the Burger's equation along with Dirichlet boundary condition is given by:
$$
\begin{equation}
    \begin{split}
        &u_t + uu_x - (0.01/\pi)u_{xx} = 0\\
        &u(0,x) = -sin(\pi x), \\
        &u(t,-1)=u(t,1)=0,
    \end{split}
\end{equation}
$$
with $x\in [-1,1]$ and $t\in [0,1]$. Let's define
$$
\begin{equation}
    f(t,x)=u_t + uu_x - (0.01/\pi)u_{xx}.
\end{equation}
$$
and approximate the field $u(t,x)$ by a deep neural network. The shared parameters between the neural networks $u(t,x)$ and $f(t,x)$ can be learned by minimizing the mean squared error loss:
$$
\begin{equation}
    MSE = MSE_u + MSE_f,
\end{equation}
$$
where
$$
\begin{equation}
    MSE_u = \frac{1}{N_u} \sum \limits^{N_u}_{i=1}|u(t^i_u,x^i_u)-u^i|^2,
\end{equation}
$$
and
$$
\begin{equation}
    MSE_f = \frac{1}{N_f}\sum \limits^{N_f}_{i=1}|f(t^i_f,x^i_f)|^2.
\end{equation}
$$
Here, $N_u$ refers to the number of data points where we have obervations of the solution $u(t,x)$ (these data points typically include initial and boundary conditions) and $N_f$ refers to the number of collocation points where the differential equation residuals are evaluated (these points are used to enforce the physical constraints dictated by the PDE).

### Discrete time models

Using neural networks alone in discrete time problems may lead to some issues:

1. Traditional PINNs require a large number of collocation points across the entire spatiotemporal domain to enforce the PDE constraints accurately.

2. PINNs can struggle with accurately capturing the temporal evolution of the solution, especially when the temporal gap between data points is large.

3. Ensuring numerical stability and efficiency in solving PDEs with traditional PINNs can be challenging, particularly for stiff equations.

4. Traditional PINNs may require significant computational resources due to the need for a large number of collocation points and the complexity of solving the PDE constraints across the entire domain.

5. In many practical scenarios, data may only be available at discrete and sparse time intervals. Without proper time-stepping methods, the neural network might struggle to interpolate and predict the solution accurately between these intervals.

To solve these issues, the traditional PINNs are applied alongside the classical family of numerical methods, known as Runge-Kutta methods. In this way, each of the issues takes benefit of the methods as following:

1. The Runge-Kutta method can be used to step through time, reducing the need for a dense grid of collocation points. By using discrete time steps, the model can handle sparse data more effectively, ensuring accurate predictions even with limited temporal data.

2. The Runge-Kutta method is known for its accuracy in solving ordinary differential equations (ODEs) through higher-order approximations. By integrating this method, the temporal dynamics of the PDE can be captured more precisely, improving the overall accuracy of the solution.

3. The Runge-Kutta method provides a structured and stable approach to time-stepping, which can help maintain numerical stability. This is especially important for stiff PDEs where explicit methods might struggle without very small time steps.

4. By discretizing the time domain and using the Runge-Kutta method, the number of required collocation points can be reduced. This approach focuses computational efforts on key time steps, potentially lowering the overall computational cost.

5. The Runge-Kutta method can efficiently interpolate the solution at intermediate time steps, ensuring that the network adheres to the PDE constraints even with sparse data.

Applying the Runge-Kutta method to $f(t,x)=u_t + \mathcal{N}[u]$, it is possible to write:
$$
\begin{equation}
    \begin{split}
        u^{n+c_i} = u^n - h \sum \limits^{q}_{j=1}a_{ij}\mathcal{N}[u^{n+c_j}],\\
        u^{n+1} = u^n - h \sum \limits^{q}_{j=1} b_j\mathcal{N}[u^{n+c_j}],
    \end{split}
\end{equation}
$$
with $i=1,...,q$ and, where $c_i$ is the intermediate stage and $h=\Delta t$. The first equation refers to the final update of the future time step and, the second equation refers to the intermediate steps. The coefficients $a_{ij}$, $b_j$ and $c_i$ refers to Runge-Kutta coefficients.  It is important to observe that in our case we are just interested in updtaing the temporal variable, keeping the spatial one fixed. Thus, each stage is given by
$$
\begin{equation}
    k_j = \mathcal{N}[u^{n+c_j}]
\end{equation}
$$


