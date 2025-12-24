import matplotlib.pyplot as plt

def plot_different_forecast_strategies_profits(vals):
    _, ax = plt.subplots()
    for key, val in vals.items():
        ax.scatter(str(key[0]) + '-' + str(key[1]), val, alpha=0.3, edgecolors='none')

    ax.legend()
    ax.grid(True)

    plt.show()


if __name__ == "__main__":
    x = {}
    x[(0,1)] = 10
    x[(0,2)] = 20
    plot_different_forecast_strategies_profits(x)
