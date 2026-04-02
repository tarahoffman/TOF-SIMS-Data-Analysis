import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Data Configuration - folder path with .txt files of fragment 3D depth profiles
# See the provided folder for an example of what this should look like.
folder_path = 'data/Baseline HC Anode'
assert os.path.exists(folder_path), (
    f"Directory does not exist!\n"
    f"You asked for: {folder_path}\n"
    f"Your current working directory is: {os.getcwd()}\n"
    f"Therefore the absolute path is: {os.path.abspath(folder_path)},"
    f"but this does not exist!\n"
    f"If your dataset exists, you probably need to change your working directory, or change the dataset path."
)

# Spatial Binning (Averaging voxels to reduce noise and computation time)
bin_x, bin_y, bin_z = 16, 16, 10

# Data loading and volume reconstruction
image_data = {}

# Iterate through all .txt files in teh specified folder, reconsutrct each fragment's
# 3D intensity volume, and cache it as a binary (.npy) file for faster future loading.
# Fragment labels are extracted from filenames and used as keys for storing volumes.
for filename in os.listdir(folder_path):
    # Only process text files
    if filename.endswith(".txt"):
        # Extract fragment label (text after the last " - ")
        fragment_label = os.path.splitext(filename)[0].split(" - ")[-1]

        file_path = os.path.join(folder_path, filename)
        # Define the path for the binary cache file
        np_file_path = file_path.replace(".txt", ".npy")

        # Check if the binary version already exists
        if os.path.exists(np_file_path):
            # Load binary data (Very fast)
            volume = np.load(np_file_path)
        else:
            # Load raw ASCII data (Slow)
            print(f"Converting {filename} to binary format...")
            data = np.loadtxt(file_path, comments="#")

            # Determine grid dimensions
            nx = int(data[-1, 0] + 1)
            ny = int(data[-1, 1] + 1)
            nz = int(data[-1, 2] + 1)

            # Reshape to 3D volume (Fortran order for TOF-SIMS spatial indexing)
            volume = data[:, 3].reshape((nx, ny, nz), order="F")

            # Save as .npy for future use
            np.save(np_file_path, volume)

        image_data[fragment_label] = volume

# Align and stack all fragment volumes into a 4D array [Fragment, X, Y, Z]
intensity_names = list(image_data.keys())
intensity_volumes = np.stack(list(image_data.values()))

# Spatial Binning/Down sampling for multivariate stability
# The 3D volumes are reshaped into blocks of size (bin_x, bin_y, bin_z),
# and the mean intensity is taken within each block
n_frag, x, y, z = intensity_volumes.shape
binned_data = intensity_volumes.reshape(
    n_frag,
    x // bin_x, bin_x, # splits the x dimension into "x / bin_x" bins of size "bin_x"
    y // bin_y, bin_y,
    z // bin_z, bin_z
)
binned_data = binned_data.mean(axis=(2, 4, 6)) # note 2 corresponds to bin_x, 4 corresponds to bin_y, etc.

# Extract the new spatial dimensions (X, Y, Z), excluding fragment axis
new_x, new_y, new_z = binned_data.shape[1:]

# Flatten spatial dimensions into a feature matrix because PCA expects 2d data
# Rows = voxels (samples), Columns = chemical fragments (features)
flattened_data = np.reshape(binned_data, (n_frag, -1)).T

# PCA
pca = PCA(n_components=2, random_state=42)
pc_scores = pca.fit_transform(flattened_data)

# Extract Loadings and print summary
loadings = pca.components_

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY: PRINCIPAL COMPONENT ANALYSIS")
print("=" * 60)

for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i + 1}: {ratio * 100:.2f}% Variance Explained")

print("\nCOMPONENT LOADINGS (Chemical Contributions):")
for i, component in enumerate(loadings):
    print(f"\n--- Principal Component {i + 1} ---")
    sorted_indices = np.argsort(np.abs(component))[::-1]
    print(f"{'Fragment':<20} | {'Loading':>10}")
    print("-" * 35)
    for idx in sorted_indices:
        print(f"{intensity_names[idx]:<20} | {component[idx]:>10.4f}")

print("=" * 60 + "\n")

# Plot Results
_, _, Z = np.meshgrid(np.arange(new_x), np.arange(new_y), np.arange(new_z), indexing='ij')
z_flat = Z.flatten().astype(float)

z_norm = (z_flat.max() - z_flat) / (z_flat.max() - z_flat.min())
colors = plt.cm.coolwarm(z_norm)

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(pc_scores[:, 0], pc_scores[:, 1], c=colors, s=12, alpha=0.6, edgecolors='none')

ax.set_xlabel('Principal Component 1', fontsize=14, fontweight='bold', family='arial')
ax.set_ylabel('Principal Component 2', fontsize=14, fontweight='bold', family='arial')
ax.tick_params(direction="in", width=1.5, labelsize=12)

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

plt.tight_layout()
plt.show()