import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os
import matplotlib.ticker as ticker


############## Inputs ##############


#folder path containing 3D TOF-SIMS depth profile .txt files
folder_path = 'data/Baseline HC Anode'

# Fragment you wish to analyze and plot
# Must EXACTLY match the fragment name in the file names
selected_label = ["NaF_2-"]

# Choose thickness mode:
# "nm" converts scan number to physical thickness using scan rate
# "time" keeps depth in units of sputtering time
# If you do not know the scan rate, use "time"
thickness_mode = "nm"   # options: "nm" or "time"

#Time per scan (seconds)
# Can be determined from point spacing in 1D depth profile data
scan_time = 5

# Scan rate for your given material (nm/s)
# With our TOF-SIMS measurement settings: graphite = 0.04 nm/s, layered oxide = 0.03 nm/s
scan_rate = 0.04

# Colormap used for 3D surface plots
threeDcolor = 'turbo'

# Color scale limits
# Set to None for automatic scaling per dataset
# Set to numbers for consistent scale across multiple datasets
vmin, vmax = None, None

# Apply sliding binning across x-y to improve depth profile signal at each x, y point
# I chose bin size of 20 (pixels), if you choose too low of a bin size, the depth profile at each point is too noisy
# If you go too high of a bin size, you lose x-y resolution in your resulting thickness map
# The total depth profile and the binned depth profile at a given point will be plotted
# bin size 20 creates a 237 x 237 pixel image
xy_bin_size = 20


############## Check Inputs Are Valid ##############


#Check that folder path exists
assert os.path.exists(folder_path), (
    f"Directory does not exist!\n"
    f"You asked for: {folder_path}\n"
    f"Your current working directory is: {os.getcwd()}\n"
    f"Therefore the absolute path is: {os.path.abspath(folder_path)},"
    f"but this does not exist!\n"
    f"If your dataset exists, you probably need to change your working directory, or change the dataset path."
)

# Make sure that "selected_label" exists in the folder
# Get all available fragment labels from .txt files in folder
available_labels = []
for filename in os.listdir(folder_path):
    if filename.endswith(".txt") and "_nmvalues" not in filename:
        base_name = os.path.splitext(filename)[0]
        fragment_label = base_name.split(" - ")[-1]
        available_labels.append(fragment_label)

# Remove duplicates and sort for clean display
available_labels = sorted(set(available_labels))

# Find any invalid labels the user provided
invalid_labels = [label for label in selected_label if label not in available_labels]

# Raise helpful error if any are invalid
assert len(invalid_labels) == 0, (
    f"Invalid fragment label(s): {invalid_labels}\n\n"
    f"Available fragment labels in this folder are:\n"
    f"{available_labels}\n\n"
    f"Please make sure your 'selected_label' match exactly (including capitalization and symbols)."
)

#check that thickness mode selection is valid
if thickness_mode not in ["nm", "time"]:
    raise ValueError(
        f"Invalid thickness_mode '{thickness_mode}'!\n"
        f"Please choose either 'nm' (absolute thickness) or 'time' (relative thickness)."
    )


# Dictionary to store 3D intensity data for selected fragment
image_data = {}


############## Load Data ##############


# Loop through and read all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):  # Only process .txt files
        # Extract the fragment label from the filename
        base_name = os.path.splitext(filename)[0]
        fragment_label = base_name.split(" - ")[-1]  # Extract text after " - "

        # Check if the fragment label is in the selected list
        if fragment_label in selected_label:
            # Full path to the file
            file_path = os.path.join(folder_path, filename)
            # Use .npy file if already converted (faster loading)
            np_file_path = os.path.join(folder_path, filename.replace(".txt", ".npy"))

            # check if we have already written to numpy file form
            if os.path.exists(np_file_path):
                # load the data from numpy file
                data = np.load(np_file_path)

            # otherwise load txt then save as numpy
            else:
                data = np.loadtxt(file_path, comments="#")

                # Extract dimensions (x, y, z)
                x = int(data[-1, 0] + 1)
                y = int(data[-1, 1] + 1)
                z = int(data[-1, 2] + 1)

                # Reshape into 3D array (x, y, z)
                data = data[:, 3].reshape((x, y, z), order="F")

                # Save as .npy for faster future loading
                np.save(np_file_path, data)

            # Store the 3D intensity array in the dictionary
            image_data[fragment_label] = data


############## Analysis ##############


for label, data in image_data.items():

    ####### Sanity Check: Compare Total Depth Profile with Binned Depth Profile

    # Apply sliding window binning
    # This improved signal-to-noise ratio for depth profiles at each location
    binned_data = np.lib.stride_tricks.sliding_window_view(
        data, (xy_bin_size, xy_bin_size), axis=(0, 1)
    ).sum(axis=(-2, -1))

    # Pick a random (x, y) location
    rand_x = np.random.randint(0, binned_data.shape[0])
    rand_y = np.random.randint(0, binned_data.shape[1])

    # Extract profiles
    total_depth_profile = np.sum(data, axis=(0, 1))  # global
    local_depth_profile = binned_data[rand_x, rand_y, :]  # local (binned)

    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: total summed depth profile
    axes[0].plot(total_depth_profile)
    axes[0].set_title("Total summed depth profile")
    axes[0].set_xlabel("Depth index")
    axes[0].set_ylabel("Intensity")

    # Right: local binned depth profile
    axes[1].plot(local_depth_profile)
    axes[1].set_title(f"Binned depth profile (bin size = {xy_bin_size}, x={rand_x}, y={rand_y})")
    axes[1].set_xlabel("Depth index")
    axes[1].set_ylabel("Intensity")

    plt.tight_layout()
    plt.show()


    # Compute binned data for analysis
    binned_data = np.lib.stride_tricks.sliding_window_view(data, (xy_bin_size, xy_bin_size), axis=(0,1)).sum(axis=(-2, -1))

    # smooth along depth dimension
    smoothed_binned_data = gaussian_filter1d(binned_data, sigma=2)

    # Find index of maximum intensity (peak)
    location_of_max_peaks = np.argmax(smoothed_binned_data, axis=-1)


    # Initialize array to store thickness values
    location_of_50_percent = np.zeros(smoothed_binned_data.shape[0:2])

    # Loop through each spatial location
    for x in range(smoothed_binned_data.shape[0]):
        for y in range(smoothed_binned_data.shape[1]):

            # find the peak at this location
            max_peak_value = smoothed_binned_data[x,y,location_of_max_peaks[x,y]]
            # compute half value of the peak (used as the thickness threshold)
            _50_percent_value = max_peak_value / 2

            # Create an array of differences from the 50% value at each depth point
            # This gives positive values above the threshold, negative values below
            # We will use this to find where the curve crosses the 50% level
            distances = smoothed_binned_data[x,y] - _50_percent_value
            # Initialize default "closest point" at the maximum depth
            # If no crossing occurs, it will remain at the end of the z-axis
            closest_point = distances.shape[0]

            # Find where the depth profile crosses the 50% intenstiy threshold
            for index in range(location_of_max_peaks[x,y], distances.shape[0]-1):
                if distances[index] > 0 and distances[index + 1] < 0:
                    # Perform linear interpolation to get fractional crossing point
                    # Calculate total distance between the two points the bracket the 50% threshold
                    total = distances[index] + np.abs(distances[index+1])
                    # Fractional distance of where teh 50% point lies between the two neighboring z indices
                    fraction = distances[index] / total
                    # Add fractional value to current index to get more accurate EEI thickness
                    # This accounts for the fact that the 50% point may lie between discrete depth measurements
                    closest_point = index + fraction
                    # Stop searching after the first crossing is found
                    break

            # Store the results
            location_of_50_percent[x,y] = closest_point



    # Convert depth index to physical thickness or time
    if thickness_mode == "nm":
        nm_values = location_of_50_percent * scan_time * scan_rate
    elif thickness_mode == "time":
        nm_values = location_of_50_percent * scan_time

    # Color scaling (manual vs automatic)
    if vmin is None:
        vmin_plot = np.min(nm_values)
    else:
        vmin_plot = vmin

    if vmax is None:
        vmax_plot = np.max(nm_values)
    else:
        vmax_plot = vmax


    ############## Save Output to .txt file ##############


    # Define the output file path with "_nmvalues" appended to the filename
    nm_values_file_path = os.path.join(folder_path, f"{label}_nmvalues.txt")
    # Save nm_values as a flattened list in the text file
    np.savetxt(nm_values_file_path, nm_values.flatten(), fmt="%.6f")


    ############## 2D Plot ##############


    fig, ax = plt.subplots()
    cax = ax.imshow(np.rot90(nm_values), cmap='viridis', vmax=vmax, vmin=vmin)
    # plt.colorbar()
    cbar = fig.colorbar(cax)

    # Set color bar ticks to whole numbers automatically
    cbar.locator = ticker.MaxNLocator(integer=True)
    cbar.update_ticks()

    # Remove ticks and tick labels on both axes
    plt.xticks([])  # Removes the x-axis ticks
    plt.yticks([])  # Removes the y-axis ticks
    plt.axis('off')

    # Set the font properties for the color bar ticks
    cbar.ax.tick_params(labelsize=18)

    for label in cbar.ax.get_yticklabels():
        label.set_fontname('Arial')  # Set font to Arial
        label.set_fontweight('bold')  # Make font bold

    plt.show()


############## 3D Plot ##############


# Create meshgrid for x and y coordinates
x = np.arange(nm_values.shape[0])
y = np.arange(nm_values.shape[1])
x, y = np.meshgrid(x, y)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(x, y, nm_values, cmap=threeDcolor, edgecolor='none',  vmin=vmin, vmax=vmax)

# Remove axes
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.grid(False)  # Hide grid
ax.set_facecolor('none')  # Set background to transparent

# Remove axis panes and gridlines for a clean 3D surface
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.pane.fill = False  # Remove pane background
    axis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Make axis line invisible

# Remove the axis borders (the box around the entire plot)
ax.set_axis_off()

# Adjust the viewing angle to look more from the top
ax.view_init(elev=45, azim=-45)

# Set the z-axis limits (same as color scale for comparison)
ax.set_zlim(vmin, vmax)

# Add color bar next to the plot
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
cbar.ax.tick_params(labelsize=14)  # Set font size

# Set the color bar ticks to whole numbers
cbar.locator = ticker.MaxNLocator(integer=True)
cbar.update_ticks()

# Make the color bar ticks bold and use Arial font
for label in cbar.ax.get_yticklabels():
    label.set_fontname('Arial')  # Set font to Arial
    label.set_fontweight('bold')  # Make font bold

# Show the plot
plt.show()