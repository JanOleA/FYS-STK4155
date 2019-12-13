import numpy as np

def analytical_solution(x, t):
    """ Returns an analytical solution of the heat equation:
    u_t = u_(xx), t > 0, x in [0, 1]
    u(x, 0) = sin(pi*x), 0 < x < 1
    u(0, t) = 0, u(1, t) = 0, t >= 0
    """
    return np.sin(np.pi*x)*np.exp(-np.pi**2*t)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9,6))
    x = np.linspace(0,1,100)
    for i, t in enumerate(np.linspace(0, 0.5, 6)):
        plt.subplot(241 + i)
        plt.plot(analytical_solution(x, t))
        plt.ylim((0,1.2))
        plt.title(f"t = {t}")
        plt.xlabel("x")
        plt.ylabel("u(x,t)")
    plt.show()
