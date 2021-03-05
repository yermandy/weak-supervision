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


def plot_prec_recall(fontsize=14):
    set_font_size(fontsize)
    fig = plt.gcf()
    fig.set_size_inches(9, 5)
    plt.xlabel("recall", fontsize=fontsize)
    plt.ylabel("precision", fontsize=fontsize)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor = (1.05, 0.6))
    plt.tight_layout()
    plt.savefig('results/precision_recall.png', dpi=300)
    plt.show()