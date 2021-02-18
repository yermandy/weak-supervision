import matplotlib.pyplot as plt


def plot_prec_recall(fontsize=16):
    plt.rc('font', size=fontsize)
    plt.rc('axes', titlesize=fontsize)
    plt.rc('axes', labelsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel("recall", fontsize=fontsize)
    plt.ylabel("precision", fontsize=fontsize)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/precision_recall.png', dpi=300)
    plt.show()
