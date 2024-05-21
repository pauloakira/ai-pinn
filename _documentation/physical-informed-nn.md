# Physical-Informed Neural Networks (PINNs)

The Physical-Informed Neural Networks integrate the physical laws described by PDEs directly into the neural network training process. This is done by incorporating the PDEs into the loss function, which guides the network to learn solutions that respect these laws. A key feature of PINNs is the use of automatic differentiation to compute the derivatives of the neural network output with respect to its inputs. This is crucial for ensuring that the network’s predictions satisfy the PDEs.

The loss function in PINNs consists of two main component:
- **Data misfit**: This term measures the difference between the neural network’s predictions and the available training data.
- **PDE Residuals**: This term enforces the physical laws by measuring the residuals of the PDEs when the network’s predictions are substituted into them.

The combined loss function guides the training process, ensuring that the learned solutions are consistent with both the observed data and the physical laws. The network is trained using gradient-based optimization methods (e.g., stochastic gradient descent, Adam, L-BFGS). The gradients of the loss function with respect to the network parameters are computed using automatic differentiation.

Regarding data, the Physical-Informed Neural Networks, can be of two types:

- **Data-Driven Solutions**: PINNs can be used to find solutions to PDEs when the form of the PDE is known, and the goal is to find the specific solution that fits the data. Example: Solving the Burgers' equation to find the velocity field given initial and boundary conditions.

- **Data-Driven Discovery**: PINNs can also be used to discover the form of the PDE itself from data. This involves learning the parameters of the PDE that best describe the observed data. Example: Discovering the parameters of the Navier-Stokes equations from flow data.

## Data-drive solutions (Continuous time models)

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

In one space dimension, the Burger's equation along with tDirichlet boundary condition is given by:
$$
\begin{equation}
    \begin{split}
        &u_t + uu_x - (0.01/\pi)u_{xx} = 0\\
        &u(0,x) = -sin(\pi x), \\
        &u(t,-1)=u(t,1)=0.
    \end{split}
\end{equation}
$$
Let's define
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