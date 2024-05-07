# Deep operator networks for bioheat transfer problems with parameterized laser source functions



## Abstract

To numerically characterize optothermal interactions at the laser-tissue interface during tissue imaging or surgery, the bioheat transfer partial differential equation (PDE) needs to be solved with a laser excitation source function. Considering various possible experimental scenarios, the PDEs become parametric and need to be solved for several optical and laser parameters. Classical numerical solvers that are used for performing such simulations are computationally expensive and time consuming. In this work, we use the recent deep operator network (DeepONet) framework for learning the mapping between infinite dimensional function spaces of the parameterized laser source functions and the solutions of the parameterized bioheat transfer PDE. We propose a new neural network architecture, Res-DeepONet, for training a purely physics-informed DeepONet with laser heating functions comprising of Gaussian beam profiles. We also showcase the versatility and robustness of our network architecture within purely data-driven and hybrid training frameworks for applications where labeled ground truth data may be accessible. The proposed neural network architecture exhibits excellent generalization performance on unseen Gaussian source functions with 10× higher accuracy when compared to traditional feedforward network architectures. Furthermore, zero-shot performance studies highlight the exceptional extrapolation capabilities of our network with 50× higher accuracy compared to traditional DeepONet architectures at inference times that are two orders of magnitude faster than classical solvers. 

## Res-DeepONet architecture

We take inspiration from the original ResNet, UNet, and Network-in-Network papers to formulate our network architecture. Distint architectures have been employed for the branch and trunk networks.

![image](https://github.com/adi-roy/Res-DeepONet/assets/145612549/e4dc1713-d6b8-42de-8420-75de8d63b380)


## Generalization performance of physics-informed Res-DeepONet

Our model predicts both peak temperatures and thermal diffusion characteristics with an average error of 0.17%. The figure below shows temperature contours plotted on the space-time domain for 3 inference examples (minimum error, 50% median error, and maximum error) from the set of 100 test source functions. 

![image](https://github.com/adi-roy/Res-DeepONet/assets/145612549/e35b9d12-2953-44c2-821e-52364fdeb2bc)

## Performance comparison between Res-DeepONet variants

To further understand the applicability of our network architecture on other possible variations of the general DeepONet framework, we implemented Res-DeepONet on purely data driven and hybrid frameworks as well. 

| *Res-DeepONet variant*  | *Testing loss* | *Training time (in mins)* |
| ------------- | ------------- | ------------ |
| Data driven   | 0.18%  | 12 |
| Physics-informed   | 0.25%  | 39 |
| Hybrid  | 0.19%  | 38 |
