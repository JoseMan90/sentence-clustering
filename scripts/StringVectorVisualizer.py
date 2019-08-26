import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd


def visualize(matrix, labels, title=""):
    """
    Visualizes the word matrix using TSNE algorithm
    :param matrix: word matrix
    :param labels: labels if present
    :return: None
    """
    print("Reducing sentence matrix into lower dimensions using tSNE")
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    reduced_values = pd.DataFrame(tsne_model.fit_transform(matrix))

    plt.figure(figsize=(16, 16))
    plt.scatter(reduced_values[0], reduced_values[1], cmap="gist_ncar")
    plt.title(title)
    plt.show()

    plt.figure(figsize=(16, 16))
    plt.scatter(reduced_values[0], reduced_values[1], c=labels, cmap="gist_ncar")
    plt.title(title+" with identified clusters")
    plt.show()
