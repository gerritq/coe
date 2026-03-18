import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import plotly.graph_objects as go
from scipy import stats
from sklearn.neighbors import KernelDensity
from transformers import AutoModelForCausalLM
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def load_model():
    model_name = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    print(model)

def magnitude_change():
    vector1 = torch.tensor([2.0, 0.0])
    vector2 = torch.tensor([0.0, 2.0])

    difference = vector2 - vector1
    magnitude_change = torch.norm(difference, p=2)
    print(f"Magnitude change: {magnitude_change.item()}")

    plt.figure(figsize=(5, 5))
    plt.quiver(0, 0, vector1[0], vector1[1], angles="xy", scale_units="xy", scale=1, color="tab:blue", label="vector1")
    plt.quiver(0, 0, vector2[0], vector2[1], angles="xy", scale_units="xy", scale=1, color="tab:orange", label="vector2")
    plt.plot([vector1[0], vector2[0]], [vector1[1], vector2[1]], "k--", label="difference")
    plt.scatter([vector1[0], vector2[0]], [vector1[1], vector2[1]], color=["tab:blue", "tab:orange"])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(0, 3)
    plt.ylim(0, 3)
    plt.title(f"Magnitude change: {magnitude_change.item():.4f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def length_change():
    vector1 = torch.tensor([2.0, 0.0])
    vector2 = torch.tensor([0.0, 2.0])

    length1 = torch.norm(vector1, p=2)
    length2 = torch.norm(vector2, p=2)
    length_change = length2 - length1
    print(f"Length change: {length_change.item()}")

def kde_two_vectors():
    rng = np.random.default_rng(42)
    group1 = rng.normal(loc=(2.0, 0.5, -0.5), scale=(0.4, 0.3, 0.5), size=(300, 3))
    group2 = rng.laplace(loc=(-1.5, 1.5, 0.75), scale=(0.5, 0.6, 0.4), size=(300, 3))

    g1 = group1[:, [0, 1]]
    g2 = group2[:, [0, 1]]

    kde1 = KernelDensity(bandwidth=1.0, kernel="gaussian").fit(g1)
    kde2 = KernelDensity(bandwidth=1.0, kernel="gaussian").fit(g2)

    data = np.vstack([g1, g2])
    x_min, y_min = data.min(axis=0) - 1.0
    x_max, y_max = data.max(axis=0) + 1.0
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    density1 = np.exp(kde1.score_samples(grid)).reshape(xx.shape)
    density2 = np.exp(kde2.score_samples(grid)).reshape(xx.shape)

    plt.figure(figsize=(5, 5))
    plt.contour(xx, yy, density1, levels=8, cmap="Blues")
    plt.contour(xx, yy, density2, levels=8, cmap="Oranges")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("KDE Contours for Two Groups (x vs y)")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

def pairplot_kde_groups():
    rng = np.random.default_rng(42)
    group1 = rng.normal(loc=(2.0, 0.5, -0.5), scale=(0.4, 0.3, 0.5), size=(300, 3))
    group2 = rng.laplace(loc=(-1.5, 1.5, 0.75), scale=(0.5, 0.6, 0.4), size=(300, 3))

    df1 = pd.DataFrame(group1, columns=["x", "y", "z"])
    df1["group"] = "group1"
    df2 = pd.DataFrame(group2, columns=["x", "y", "z"])
    df2["group"] = "group2"
    df = pd.concat([df1, df2], ignore_index=True)

    sns.pairplot(
        df,
        vars=["x", "y", "z"],
        hue="group",
        kind="kde",
        diag_kind="kde",
        corner=True,
        plot_kws={"levels": 8, "fill": False},
    )
    plt.show()

if __name__ == "__main__":
    # load_model()

    # magnitude_change()
    # length_change()
    # kde_two_vectors()
    pairplot_kde_groups()
