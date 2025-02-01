from typing import Optional
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


def plot_similarity_evolution_stability_analysis(
    similarities: np.ndarray,
    has_returned: Optional[np.ndarray] = None,
    is_fixed_point: Optional[np.ndarray] = None,
) -> plt.Figure:
    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(12, 4), sharey=True, squeeze=False
    )
    for i, sim in enumerate(similarities):
        if similarities[i, -1] == 1:
            ax = axes[0, 0]
        else:
            ax = axes[0, 1]
        ax.plot(sim, label=f"Trial {i + 1}")
    for ax in axes.flat:
        ax.set_xlabel("Step")
        ax.set_ylabel("similarity")
    title_returned = "Returned"
    if has_returned is not None:
        title_returned += f" ({np.mean(has_returned) * 100:.0f}%)"
    axes[0, 0].set_title(title_returned)
    title_not_returned = "Not returned"
    if has_returned is not None:
        title_not_returned += f" ({np.mean(~has_returned) * 100:.0f}%)."
    axes[0, 1].set_title(title_not_returned)
    fig_title = "Local stability analysis."
    if has_returned is not None and is_fixed_point is not None:
        fig_title += f" Perturbations converged to fixed point: {np.mean(is_fixed_point) * 100:.0f}%"
        fig_title += f" (original: {np.mean(has_returned & is_fixed_point) * 100:.0f}%, other: {np.mean(~has_returned & is_fixed_point) * 100:.0f}%)"
    fig.suptitle(fig_title)
    # fig.tight_layout()
    return fig


# def analyze_local_stability_random(
#     network, dynamics, num_flips=10, num_steps=1000, num_trials=30, seed=42
# ):
#     seeds = np.random.default_rng(seed).integers(0, 2**32, num_trials)
#     initial_state = network.state.copy()
#     similarities = []
#     for i in range(num_trials):
#         rng = np.random.default_rng(seeds[i])
#         network.set_state(initial_state.copy())
#         flip_indices = rng.choice(network.N, num_flips, replace=False)
#         network.state[flip_indices] *= -1
#
#         # we pass the unperturbed state as reference state
#         logger = Logger(reference_state=initial_state, log_interval=1)
#         stopping_condition = SimpleStoppingCondition(
#             max_iterations=num_steps, stable_steps_needed=None
#         )
#         simulation = HopfieldSimulation(network, dynamics, stopping_condition, logger)
#         simulation.run()
#         similarities.append(logger.similarity_history)
#     return np.array(similarities)
