from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

def plot_TSNE(data, n_components=2):
    print(sklearn.__version__ >= "1.0.2")
    
    result = TSNE(
        n_components=n_components, 
        learning_rate='auto', 
        init='random'
    ).fit_transform(data)
    
    plt.scatter(result[:, 0], result[:, 1], cmap=plt.cm.Spectral)
    plt.title("T-SNE")
    plt.axis("tight")

    plt.show()
