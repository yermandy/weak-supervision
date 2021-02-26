import matplotlib.pyplot as plt


def set_font_size(fontsize):
    params = {
        'font.size': fontsize,
        'axes.titlesize': fontsize,
        'axes.labelsize': fontsize,
    }
    plt.rcParams.update(params)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)


def plot_prec_recall(fontsize=16):
    set_font_size(fontsize)
    plt.xlabel("recall", fontsize=fontsize)
    plt.ylabel("precision", fontsize=fontsize)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/precision_recall.png', dpi=300)
    plt.show()