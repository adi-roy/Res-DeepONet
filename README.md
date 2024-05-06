# Deep operator networks for bioheat transfer problems with parameterized laser source functions

Aditya Roy, Andrew Duplissis, Biswajit Mishra, Adela Ben-Yakar

## Abstract

To numerically characterize optothermal interactions at the laser-tissue interface during tissue imaging or surgery, the bioheat transfer partial differential equation (PDE) needs to be solved with a laser excitation source function. Considering various possible experimental scenarios, the PDEs become parametric and need to be solved for several optical and laser parameters. Classical numerical solvers that are used for performing such simulations are computationally expensive and time consuming. In this work, we use the recent deep operator network (DeepONet) framework for learning the mapping between infinite dimensional function spaces of the parameterized laser source functions and the solutions of the parameterized bioheat transfer PDE. We propose a new neural network architecture, Res-DeepONet, for training a purely physics-informed DeepONet with laser heating functions comprising of Gaussian beam profiles. We also showcase the versatility and robustness of our network architecture within purely data-driven and hybrid training frameworks for applications where labeled ground truth data may be accessible. The proposed neural network architecture exhibits excellent generalization performance on unseen Gaussian source functions with 10× higher accuracy when compared to traditional feedforward network architectures. Furthermore, zero-shot performance studies highlight the exceptional extrapolation capabilities of our network with 50× higher accuracy compared to traditional DeepONet architectures at inference times that are two orders of magnitude faster than classical solvers. 

## Generalization performance of physics-informed Res-DeepONet

Our model predicts both peak temperatures and thermal diffusion characteristics with an average error of 0.17%. The figure below shows temperature contours plotted on the space-time domain for 3 inference examples (minimum error, 50% median error, and maximum error) from the set of 100 test source functions. 


