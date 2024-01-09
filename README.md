# Efficient-Learning-Long-Range-QS
Code used for carrying out the simulations in the experimental section of the paper Efficient Learning of Long-Range and Equivariant Quantum Systems, Štěpán Šmíd and Roberto Bondesan, available at https://arxiv.org/abs/2312.17019 .


In this paper, we look into learning ground state properties of parametrised Hamiltonians with exponentially or polynomially decaying interactions within the same topological phase using a number of training samples that scales logarithmically with the system size to obtain a guaranteed average additive error. Large systems are simulated using tensor network methods and DMRG. In the case of equivariant systems, such as a system on a periodic chain, we obtain a further reduction to a constant number of samples. 


This Python library requires NumPy, SciPy, Matplotlib, scikit-learn and TeNpy. For a simple installation, a Conda environment file LongRangeQS.yml is included.


The main file main.py specifies the model, boundary conditions, and many other options. models.py includes the MPO implementations of the Heisenberg and Ising chain used, observables.py includes calculations of the specified observable using exact diagonalisation or DMRG (and the options for DMRG), and finally lasso.py includes the specific feature mapping and lasso models used for the machine learning.
