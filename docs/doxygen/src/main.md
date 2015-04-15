PetaVision   {#mainpage}
==========

PetaVision is an open source, object oriented neural simulation toolbox optimized for high-performance multi-core, multi-node computer architectures.  PetaVision is intended for computational neuroscientists who seek to apply neuromorphic models to hard signal processing problems; both to improve on the performance of existing algorithms and/or to gain insight into the computational mechanisms underlying biological neural processing.

Most of the low-level details relating to high-performance execution, such as calls to MPI and CUDA libraries,  are implemented at the superclass level, allowing scientific programmers who lack expertise in high-performance language constructs to add new functionality (i.e. new learning rules, channel types, etc) without sacrificing the high-performance capabilities of the underlying simulation engine. 

Currently, the PetaVision library contains classes for implementing both spiking and non-spiking neurons, ranging from conductance-based Leaky-Integrate-and-Fire (LIF) elements with both chemical and electrical synaptic inputs to very simple input/output elements suitable for constructing conventional Artificial Neural Networks (ANNs).  PetaVision also allows for both shared-weight and unique-weight synaptic connections, which can be modified by customizable plasticity rules; currently Hebbian and STDP models are implemented.  


- [About Us](src/about_us.md)
- Installation
    -[AWS Install](src/install_aws.md)
    -[OSX Install](src/install_osx.md)
    -[Ubuntu Install](src/install_ubuntu.md)


