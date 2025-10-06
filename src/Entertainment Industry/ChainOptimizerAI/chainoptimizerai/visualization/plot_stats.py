import matplotlib.pyplot as plt


def plot_costs(costs):
    plt.figure()
    plt.plot(costs, label='Total Cost')
    plt.xlabel('Step')
    plt.ylabel('Cost')
    plt.title('Costs over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()


