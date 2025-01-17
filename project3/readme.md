## Project 3, Solving differential equations using neural networks

GitHub repository: https://github.com/JanOleA/FYS-STK4155/tree/master/project3

All code files are in the root folder, project3. Figures are in a subfolder called "figures". The report is also in the root folder.

Main programs list:
- analytical.py     - Simple file containing the analytical solution. Chose to have it in a separate file as it is reused in many of the other files.
- finite_diff.py    - Contains code for running the explicit finite difference scheme for the PDE.
- nn_solver.py      - Contains code for solving the PDE using a TensorFlow deep neural network.
- eigenvalue.py     - Contains code for computing eigenpairs using a TensorFlow DNN, as well as standard linear algebra methods using Numpy.
- resources.py      - Contains own MSE and R2 score functions.

Note that some of the heavy computations were done on the UiO ML2 server, which Markus Leira Asprusten helped me with, so the commits of some of the results are shown as being done by him. This only applies to some of the figures and data files that resulted from me running the code on ML2.
