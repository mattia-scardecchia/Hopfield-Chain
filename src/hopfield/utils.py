import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def visualize_state(state: np.ndarray) -> plt.Figure:
    N = len(state)
    n = int(np.ceil(np.sqrt(N)))
    padded_state = np.pad(state, (0, n * n - N), mode="constant", constant_values=0)
    padded_state = padded_state.reshape((n, n))
    cmap = mcolors.ListedColormap(["black", "gray", "white"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(padded_state, cmap=cmap, norm=norm)
    plt.title("Network State")
    return fig


def plot_couplings_histogram(J: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(J.flatten(), bins=20, edgecolor="black")
    ax.set_xlabel("Coupling Strength")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Coupling Strengths")
    ax.grid(True, linestyle="--", alpha=0.6)
    return fig


def plot_energy_pairs_histogram(energies: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(energies.flatten(), bins=20, edgecolor="black")
    ax.set_xlabel("Energy")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Energy Pairs")
    ax.grid(True, linestyle="--", alpha=0.6)
    return fig
