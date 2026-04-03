import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math

############## Inputs ##############


# Folder path containing _nmvalues.txt files
folder_path = 'data/Baseline HC Anode'

# Dictionary to store the data for each fragment
image_data = {}


############## Load Data ##############


# Loop through and read all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".txt") and "nmvalues" in filename:  # Only process .txt files
        # Extract the fragment label from the filename
        base_name = os.path.splitext(filename)[0]
        # Extract a clean fragment label (remove "_nmvalues")
        fragment_label = base_name.replace("_nmvalues", "")

        # Full path to the file
        file_path = os.path.join(folder_path, filename)
        # Load the file, ignoring header lines starting with '#'
        data = np.loadtxt(file_path, comments="#")
        # Store the 3D intensity array in the dictionary
        image_data[fragment_label] = data


############## PCA ##############


# Each fragment now has a 2D array from sliding binning
# We want to stack them along a new axis so that each x,y point has a vector of thickness values
labels = image_data.keys() # Save fragment names for labeling later
image_data = np.stack(list(image_data.values()), axis=-1)

# do PCA with 1 principal component
pca = PCA(n_components=1, random_state=42)

# Fit PCA to the image data and transform it to the latent space
# latent space will have one value per x,y point representing the PC1
latent_space = pca.fit_transform(image_data)

#plot PCA scores from the origin (so that scores are all positive)
mean_in_pca_space = (pca.mean_).dot(pca.components_.T)
latent_space = latent_space + mean_in_pca_space

# Get how much variance is explained by the PC
explained_variance = pca.explained_variance_ratio_

# Print the explained variance result
print("Explained variance ratio for each component:", explained_variance)
print(f"Total variance explained by PC1: {explained_variance[0]*100:.2f}%")


# Extract loadings for each component
loadings = pca.components_  # shape: (n_components, n_fragments)

# Convert to a labeled dictionary
fragment_labels = list(labels)
for i, component in enumerate(loadings):
    print(f"\nPCA Component {i+1} Loadings:")
    for frag, value in zip(fragment_labels, component):
        print(f"  {frag:15s} {value:.3f}")

# Simple metric for each fragment's relative contribution PC
loadings = pca.components_[0]
relative_contrib = (loadings**2) / np.sum(loadings**2)
# Print PC loadings
for frag, contrib in zip(fragment_labels, relative_contrib):
    print(f"{frag:15s} contributes {contrib*100:.2f}% to PC1")


# Assume image square
new_size = int(math.sqrt(latent_space.shape[0]))
latent_space = latent_space.flatten()
latent_space = latent_space.reshape(new_size,new_size)


############## Plot ##############


fig, ax = plt.subplots(figsize=(6, 6))

# Define physical axes in µm
size_um = 100  # total size in µm
num_pixels = latent_space.shape[0]  # 237
pixel_size = size_um / num_pixels   # µm per pixel

plt.imshow(np.flipud(np.fliplr(latent_space)), cmap='inferno', origin='lower',
           extent=[0, size_um, 0, size_um], vmin=5, vmax=43)  # set physical axes


cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.set_label('PC Score', fontsize=20, fontname='arial', fontweight='bold')

cbar.ax.tick_params(labelsize=18)

for label in cbar.ax.get_yticklabels():
    label.set_fontname('arial')
    label.set_fontweight('bold')

plt.xlabel('X (µm)', fontdict={'fontsize': 24, 'fontname': 'arial', 'fontweight': 'bold'})
plt.ylabel('Y (µm)', fontdict={'fontsize': 24, 'fontname': 'arial', 'fontweight': 'bold'})
ax.tick_params(axis="x", direction="in", labelsize=18)
ax.tick_params(axis="y", direction="in", labelsize=18)
for label in ax.get_xticklabels():
    label.set_fontweight('bold')
for label in ax.get_yticklabels():
    label.set_fontweight('bold')
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

plt.tight_layout()
plt.show()