# Physical-Informed Neural Networks (PINNs)

The Physical-Informed Neural Networks integrate the physical laws described by PDEs directly into the neural network training process. This is done by incorporating the PDEs into the loss function, which guides the network to learn solutions that respect these laws. A key feature of PINNs is the use of automatic differentiation to compute the derivatives of the neural network output with respect to its inputs. This is crucial for ensuring that the network’s predictions satisfy the PDEs.

The loss function in PINNs consists of two main component:
- **Data misfit**: This term measures the difference between the neural network’s predictions and the available training data.
- **PDE Residuals**: This term enforces the physical laws by measuring the residuals of the PDEs when the network’s predictions are substituted into them.

The combined loss function guides the training process, ensuring that the learned solutions are consistent with both the observed data and the physical laws. The network is trained using gradient-based optimization methods (e.g., stochastic gradient descent, Adam, L-BFGS). The gradients of the loss function with respect to the network parameters are computed using automatic differentiation.

Regarding data, the Physical-Informed Neural Networks, can be of two types:

- **Data-Driven Solutions**: PINNs can be used to find solutions to PDEs when the form of the PDE is known, and the goal is to find the specific solution that fits the data. Example: Solving the Burgers' equation to find the velocity field given initial and boundary conditions.

- **Data-Driven Discovery**: PINNs can also be used to discover the form of the PDE itself from data. This involves learning the parameters of the PDE that best describe the observed data. Example: Discovering the parameters of the Navier-Stokes equations from flow data.