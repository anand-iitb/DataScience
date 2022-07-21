import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class MyPCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        """
        Assumes observations in X are passed as rows of a numpy array.
        """

        # Translate the dataset so it's centered around 0
        translated_X = X - np.mean(X, axis=0)

        # Calculate the eigenvalues and eigenvectors of the covariance matrix
        e_values, e_vectors = np.linalg.eigh(np.cov(translated_X.T))

        # Sort eigenvalues and their eigenvectors in descending order
        e_ind_order = np.flip(e_values.argsort())
        e_values = e_values[e_ind_order]
        e_vectors = e_vectors[e_ind_order]

        # Save the first n_components eigenvectors as principal components
        principal_components = np.take(e_vectors, np.arange(self.n_components), axis=0)

        return np.matmul(translated_X, principal_components.T)

def plot_pca_results(pca_class, dataset, plot_title):
    X = dataset.data
    y = dataset.target
    y_names = dataset.target_names

    pca = pca_class(n_components=1)
    B = pca.fit_transform(X)
    B = np.concatenate([B, np.zeros_like(B)], 1)

    scatter = plt.scatter(B[:, 0], B[:, 1], c=y)
    scatter_objects, _ = scatter.legend_elements()
    plt.title(plot_title)
    plt.legend(scatter_objects, y_names, loc="lower left", title="Classes")
    plt.show()


dataset = datasets.load_iris()
plot_pca_results(MyPCA, dataset, "Iris - my PCA")
plot_pca_results(PCA, dataset, "Iris - Sklearn")
