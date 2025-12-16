import numpy as np
import matplotlib.pyplot as plt
from sklearn.random_projection import GaussianRandomProjection
from sklearn.datasets import make_blobs
from random_projection import RandomProjection

def sample_distances(X, X_proj, pairs):
    d0, d1 = [], []
    for i, j in pairs:
        if i == j:
            continue
        orig = np.linalg.norm(X[i] - X[j])
        proj = np.linalg.norm(X_proj[i] - X_proj[j])
        if orig > 1e-10:
            d0.append(orig)
            d1.append(proj)
    return np.array(d0), np.array(d1)

def sample_pairs(n_samples, n_pairs=2000, seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_samples, size=(n_pairs, 2))

def avg_distortion(X, X_proj, n_pairs=2000):
    pairs = sample_pairs(len(X), n_pairs)
    d0, d1 = sample_distances(X, X_proj, pairs)
    return np.mean(np.abs(d1 / d0 - 1))

X, _ = make_blobs(n_samples=1000, n_features=50, centers=10, cluster_std=3.0, random_state=42)
n_components_list = [2, 5, 10, 20]

custom_distortions = []
sklearn_distortions = []

fig, axes = plt.subplots(len(n_components_list), 2, figsize=(14, 4*len(n_components_list)))
if len(n_components_list) == 1:
    axes = np.expand_dims(axes, axis=0)

for row_idx, n_components in enumerate(n_components_list):
    custom_rp = RandomProjection(n_components=n_components, random_state=42)
    X_custom = custom_rp.fit_transform(X)

    sk_rp = GaussianRandomProjection(n_components=n_components, random_state=42)
    X_sklearn = sk_rp.fit_transform(X)

    custom_dist = avg_distortion(X, X_custom)
    sklearn_dist = avg_distortion(X, X_sklearn)
    custom_distortions.append(custom_dist)
    sklearn_distortions.append(sklearn_dist)
    print(f"n_components={n_components} | Custom avg distortion: {custom_dist:.4f} | Sklearn avg distortion: {sklearn_dist:.4f}")

    pairs = sample_pairs(len(X))
    d0_c, d1_c = sample_distances(X, X_custom, pairs)
    d0_s, d1_s = sample_distances(X, X_sklearn, pairs)
    max_d = max(d0_c.max(), d0_s.max())

    ax_scatter = axes[row_idx, 0]
    ax_scatter.scatter(d0_c, d1_c, s=8, alpha=0.5, label="Custom RP")
    ax_scatter.scatter(d0_s, d1_s, s=8, alpha=0.5, label="Sklearn RP")
    ax_scatter.plot([0, max_d], [0, max_d], linestyle="--", color="black")
    ax_scatter.set_title(f"Distance Preservation (n={n_components})")
    ax_scatter.set_xlabel("Original distance")
    ax_scatter.set_ylabel("Projected distance")
    ax_scatter.legend()

    ax_hist = axes[row_idx, 1]
    ratios_c = d1_c / d0_c
    ratios_s = d1_s / d0_s
    ax_hist.hist(ratios_c, bins=50, alpha=0.6, label="Custom RP")
    ax_hist.hist(ratios_s, bins=50, alpha=0.6, label="Sklearn RP")
    ax_hist.axvline(1.0, linestyle="--", color="black")
    ax_hist.set_title(f"Distance Ratios Histogram (n={n_components})")
    ax_hist.set_xlabel("Projected / Original distance")
    ax_hist.set_ylabel("Count")
    ax_hist.legend()

plt.tight_layout()
plt.show()

custom_rp_2d = RandomProjection(n_components=2, random_state=42)
X_custom_2d = custom_rp_2d.fit_transform(X)
sk_rp_2d = GaussianRandomProjection(n_components=2, random_state=42)
X_sklearn_2d = sk_rp_2d.fit_transform(X)

plt.figure(figsize=(7,5))
plt.scatter(X_custom_2d[:,0], X_custom_2d[:,1], s=10, alpha=0.6, label="Custom RP")
plt.scatter(X_sklearn_2d[:,0], X_sklearn_2d[:,1], s=10, alpha=0.6, label="Sklearn RP")
plt.title("2D Projection Scatter (n_components=2)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.show()

plt.figure(figsize=(7,5))
plt.plot(n_components_list, custom_distortions, marker='o', label="Custom RP")
plt.plot(n_components_list, sklearn_distortions, marker='s', label="Sklearn RP")
plt.xlabel("Number of components")
plt.ylabel("Average distortion")
plt.title("Distortion vs Number of Components")
plt.legend()
plt.grid(True)
plt.show()
