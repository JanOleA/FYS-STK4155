from analytical import analytical_solution
import numpy as np
import matplotlib.pyplot as plt
import sys
from resources import MSE, R2


def solve(dx, T):
    """ Solver for a specific case of the heat equation:
    u_t = u_(xx), t > 0, x in [0, 1]
    u(x, 0) = sin(pi*x), 0 < x < 1
    u(0, t) = 0, u(1, t) = 0, t >= 0

    Input:
    dx : spatial difference
    T  : end point of simulation

    dt is calculated from dx using the stability criterion: dt/dx^2 <= 0.5

    Returns:
    u_array : n x i, 2D array containing solution, where n is timestep,
              i is position
    t_array : 1D time array
    """

    dt = 0.5*dx**2
    N = int(np.ceil(T/dt))
    N_x = (1/dx + 1)

    C = dt/dx**2 #0.5

    if N_x%1 != 0:
        print("Must be possible to make integer number of spatial points")
        print(f"Was given dx = {dx} which gives N_x = {N_x}")
        sys.exit(1)

    N_x = int(N_x)

    u_array = np.zeros((N, N_x))
    t_array = np.zeros(N)

    x = np.linspace(0, 1, N_x)
    u_array[0] = np.sin(np.pi*x)

    for n in range(N-1):
        u_array[n + 1, 1:-1] = (C*(u_array[n, 2:]
                                - 2*u_array[n, 1:-1]
                                + u_array[n, 0:-2])
                                + u_array[n, 1:-1])

        t_array[n + 1] = t_array[n] + dt

    return u_array, t_array


if __name__ == "__main__":
    dx_list = [0.1, 0.01]

    for dx in dx_list:
        u_array, t_array = solve(dx, 1)
        plt.figure(figsize=(9,6))

        N_x = 1/dx + 1
        x = np.linspace(0, 1, int(N_x))

        times = []
        num_solutions = []
        ana_solutions = []

        N_t = len(t_array)
        for i, j in enumerate(np.linspace(0, N_t-1, 8)):
            j = int(j)
            num_solution = u_array[j]
            t = t_array[j]
            ana_solution = analytical_solution(x, t)

            if i == 1 or i == 7:
                # store solutions for two different times
                times.append(t)
                num_solutions.append(num_solution)
                ana_solutions.append(ana_solution)

            plt.subplot(241 + i)
            plt.plot(x, num_solution, label = "numerical solution")
            plt.plot(x, ana_solution, label = "analytical solution")
            plt.ylim((0,1.2))
            plt.title(f"t = {t:.3f}, dx = {dx}")
            plt.xlabel("x")
            plt.ylabel("u(x,t)")
            plt.legend()

        print(f"#################### dx = {dx} ####################")
        print(f"MSE at t = {times[0]} = {MSE(num_solutions[0], ana_solutions[0])}")
        print(f"MSE at t = {times[1]} = {MSE(num_solutions[1], ana_solutions[1])}")

        print(f"R2 at t = {times[0]}  = {R2(num_solutions[0], ana_solutions[0])}")
        print(f"R2 at t = {times[1]}  = {R2(num_solutions[1], ana_solutions[1])}\n")

    plt.show()
