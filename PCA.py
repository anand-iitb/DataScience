import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class MyPCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        # Compute the mean of the data
        self.mean = np.mean(X, axis=0)
        # Subtract the mean from the data and divide by the standard deviation
        X_centered = (X - self.mean)/np.std(X, axis=0)
        # Compute the covariance matrix
        self.covariance = np.dot(X_centered.T, X_centered) / (X.shape[0] - 1)
        # Compute the eigenvectors and eigenvalues of the covariance matrix
        self.eigvals, self.eigvecs = np.linalg.eigh(self.covariance)
        # Sort the eigenvectors and eigenvalues in descending order of the eigenvalues
        self.eigvecs = self.eigvecs[:, np.argsort(self.eigvals)[::-1]]
        self.eigvals = self.eigvals[np.argsort(self.eigvals)[::-1]]
        # Select the first n_components eigenvectors and eigenvalues
        self.eigvals = self.eigvals[:self.n_components]
        self.eigvecs = self.eigvecs[:, :self.n_components]
        # Return the transformed data
        return self.transform(X)

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.eigvecs)


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
plot_pca_results(MyPCA, dataset, "PCA on Iris")
