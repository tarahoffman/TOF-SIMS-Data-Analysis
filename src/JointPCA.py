import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


############## Inputs ##############

# Folder paths containing TOF-SIMS 3D depth profile for each fragment
folder_paths = [
    '../data/TOF-SIMS/TOF-SIMS Methods Paper/Karlas Crossover Data/3D Fragment Maps/to 300 depth/Gr',
    '../data/TOF-SIMS/TOF-SIMS Methods Paper/Karlas Crossover Data/3D Fragment Maps/to 300 depth/Li',
    '../data/TOF-SIMS/TOF-SIMS Methods Paper/Karlas Crossover Data/3D Fragment Maps/to 300 depth/SiOx'
]

# names for each sample, must match folder_paths order
sample_names = [
    "Gr",
    "Li",
    "SiOx"
]
#colors for each sample centroid & ellipse
centroid_colors = ['tab:blue', 'tab:red', 'tab:green']


assert len(folder_paths) == len(sample_names), f"You should have the same number of sample names and folder paths. Got {len(folder_paths)=} and {len(sample_names)=}"

# number of bins in x,y,z directions
bx, by, bz = 16, 16, 10


############## Load fragment maps and bin data ##############


all_samples_data = []
intensity_names = None

# Iterate through all .txt files in teh specified folder, reconsutrct each fragment's
# 3D intensity volume, and cache it as a binary (.npy) file for faster future loading.
# Fragment labels are extracted from filenames and used as keys for storing volumes.
for folder_path in folder_paths:
    sample_data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            base_name = os.path.splitext(filename)[0]
            fragment_label = base_name.split(" - ")[-1]

            file_path = os.path.join(folder_path, filename)
            np_file_path = file_path.replace(".txt", ".npy")

            if os.path.exists(np_file_path):
                data = np.load(np_file_path)
            else:
                data = np.loadtxt(file_path, comments="#")
                x, y, z = int(data[-1, 0]+1), int(data[-1, 1]+1), int(data[-1, 2]+1)
                data = data[:, 3].reshape((x, y, z), order="F")
                np.save(np_file_path, data)

            sample_data[fragment_label] = data

    # convert to stacked array
    names, values = zip(*sample_data.items())
    if intensity_names is None:
        intensity_names = names  # store consistent ordering

    # ensure common shape
    min_shape = np.min([a.shape for a in values], axis=0)
    values = [a[tuple(slice(0, m) for m in min_shape)] for a in values]
    values = np.stack(values)

    # bin
    x, y, z = values.shape[1:]
    # Check that the bin sizes evenly divide the data
    if (x % bx != 0) or (y % by != 0) or (z % bz != 0):
        raise ValueError(
            f"Invalid bin size!\n"
            f"Your data dimensions are: x={x}, y={y}, z={z}\n"
            f"Your chosen bin sizes are: bx={bx}, by={by}, bz={bz}\n"
            f"Each bin size must evenly divide the corresponding data dimension.\n"
            f"Please choose bin sizes that are factors of the data dimensions."
        )
    new_x, new_y, new_z = x // bx, y // by, z // bz

    values = values.reshape(values.shape[0], x//bx, bx, y//by, by, z//bz, bz).mean(axis=(2,4,6))

    # reshape to (channels, points)
    values = values.reshape(values.shape[0], -1)
    all_samples_data.append(values)

# ---- Combine all points from all samples for PCA ----
combined_data = np.concatenate(all_samples_data, axis=1).T  # shape: (total_points, channels)


############## Apply PCA & Print Summary ##############


pca = PCA(n_components=2, random_state=42)
pca_scores = pca.fit_transform(combined_data)  # shape: (total_points, 2)


# ---- Compute centroids for each sample ----
centroids = []
sample_score_sets = []


start = 0
for sample_values in all_samples_data:
    n_points = sample_values.shape[1]
    sample_scores = pca_scores[start:start+n_points]
    start += n_points

    centroid = sample_scores.mean(axis=0)  # mean along all points
    centroids.append(centroid)
    sample_score_sets.append(sample_scores)


centroids = np.array(centroids)  # shape: (n_samples, 2)


# Print PCA stats
print("="*60)
print("PCA SUMMARY")
print("="*60)
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: Explained Variance Ratio = {var*100:.2f}% (Eigenvalue = {pca.explained_variance_[i]:.4f})")

print("\nComponent Loadings (raw coefficients):")
for i, component in enumerate(pca.components_):
    print(f"\n--- PC{i+1} ---")
    for name, loading in zip(intensity_names, component):
        print(f"{name:20s} {loading: .4f}")
print("="*60 + "\n")

# Determine limits across all samples
all_scores_x = []
all_scores_y = []
start = 0

for i, sample_values in enumerate(all_samples_data):
    n_points = sample_values.shape[1]
    sample_scores = pca_scores[start:start+n_points]
    start += n_points
    all_scores_x.append(sample_scores[:,0])
    all_scores_y.append(sample_scores[:,1])

# Flatten to get global limits
x_min, x_max = np.min(np.concatenate(all_scores_x)), np.max(np.concatenate(all_scores_x))
y_min, y_max = np.min(np.concatenate(all_scores_y)), np.max(np.concatenate(all_scores_y))


############## Plot each sample using the common principal components ##############


start = 0
for i, sample_values in enumerate(all_samples_data):
    n_points = sample_values.shape[1]
    sample_scores = pca_scores[start:start+n_points]
    start += n_points

    PCAx, PCAy = sample_scores[:,0], sample_scores[:,1]

    # Color by depth (Z): red = surface, blue = bulk
    X, Y, Z = np.meshgrid(
        np.arange(new_x),
        np.arange(new_y),
        np.arange(new_z),
        indexing='ij'
    )

    Z_flat = Z.flatten().astype(float)
    # normalize Z from 0 (surface) → 1 (bulk)
    Z_norm = (Z_flat - Z_flat.min()) / (Z_flat.max() - Z_flat.min())
    # invert so surface = red, bulk = blue
    Z_norm = 1 - Z_norm

    colors = plt.cm.coolwarm(Z_norm)

    fig, ax = plt.subplots()
    ax.scatter(PCAx, PCAy, c=colors, s=10)
    ax.set_xlabel('PC 1', fontdict={'fontsize': 18, 'fontweight':'bold'})
    ax.set_ylabel('PC 2', fontdict={'fontsize': 18, 'fontweight':'bold'})
    ax.tick_params(axis="x", direction="in", labelsize=16)
    ax.tick_params(axis="y", direction="in", labelsize=16)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)  # change width
    plt.tight_layout()
    plt.show()


############## Plot centroids of each sample's PC scores ##############


fig, ax = plt.subplots()
for i, centroid in enumerate(centroids):
    ax.scatter(centroid[0], centroid[1], c=centroid_colors[i], s=100, marker='X')


# 95% confidence scaling factor for 2D Gaussian (chi-square, df=2)
chi2_val = 5.991  # standard value for 95%

for i, scores in enumerate(sample_score_sets):
    # covariance matrix of PCA scores
    cov = np.cov(scores, rowvar=False)

    # eigenvalues and eigenvectors
    vals, vecs = np.linalg.eigh(cov)

    # sort eigenvalues (largest first)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # angle of ellipse
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    # ellipse width and height
    width = 2 * np.sqrt(vals[0] * chi2_val)
    height = 2 * np.sqrt(vals[1] * chi2_val)

    ellipse = Ellipse(xy=centroids[i], width=width, height=height, angle=angle, edgecolor=centroid_colors[i],
        facecolor='none', linewidth=2, alpha=0.8)

    ax.add_patch(ellipse)


# Add labels
for i, name in enumerate(sample_names):
    ax.text(centroids[i, 0] + 0.02, centroids[i, 1] + 0.02, name, fontsize=16, fontweight='bold', color=centroid_colors[i])

ax.set_xlabel('PC 1', fontdict={'fontsize': 18, 'fontweight':'bold'})
ax.set_ylabel('PC 2', fontdict={'fontsize': 18, 'fontweight':'bold'})


ax.tick_params(axis="x", direction="in", labelsize=16)
ax.tick_params(axis="y", direction="in", labelsize=16)
for label in ax.get_xticklabels():
    label.set_fontweight('bold')
for label in ax.get_yticklabels():
    label.set_fontweight('bold')
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(1.5)  # change width

plt.tight_layout()
plt.show()
