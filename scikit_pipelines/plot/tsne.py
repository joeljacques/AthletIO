import matplotlib.pyplot as plt
from sklearn import manifold
from matplotlib.ticker import NullFormatter

def plot_tsne(X, y, perplexities=None, show=True, save=False, path=None, filename=None):
    # t-distributed stochastic neighbour embedding
    # for data exploration
    # expects X to be 2 dimensional (num_samples, num_features)

    if not perplexities:
        perplexities = [5, 10, 50, 100]
    n_components = 3
    (fig, subplots) = plt.subplots(1, len(perplexities), figsize=(15, 3))

    # TODO adapt for more classes

    red = y == 0
    green = y == 1

    for i, perplexity in enumerate(perplexities):
        # ax = subplots[0][i + 1]
        ax = subplots[i]

        tsne = manifold.TSNE(
            n_components=n_components,
            init="random",
            random_state=0,
            perplexity=perplexity,
            learning_rate="auto",
            n_iter=300,
        )

        Y = tsne.fit_transform(X, y)

        ax.set_title("Perplexity=%d" % perplexity)
        ax.scatter(Y[red, 0], Y[red, 1], c="r")
        ax.scatter(Y[green, 0], Y[green, 1], c="g")
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis("tight")

    if show:
        plt.show()
    if save:
        pass
        # _save(fig, "t-SNE", save, path, filename)
