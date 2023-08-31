import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from packaging import version
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

assert version.parse(tf.__version__) >= version.parse("2.8.0")
print("Tensorflow version: ", tf.__version__)

# define the default font sizes to make the figures prettier
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

# create the images/ann folder
from pathlib import Path
IMAGES_PATH = Path() / "images" / "ann"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


"""
From Biological to Artificial Neurons
"""

# The Perceptron
iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 0)

per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)

X_new = [[2, 0.5], [3,1]]
y_pred = per_clf.predict(X_new)
print("Result: ", y_pred)



