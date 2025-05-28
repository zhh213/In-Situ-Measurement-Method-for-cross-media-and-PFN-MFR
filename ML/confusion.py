import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Data
confusion_data = [
    [[85, 23], [16, 76]],
    [[104, 4], [2, 90]],
    [[108, 0], [3, 89]],
    [[76, 32], [23, 69]],
    [[74, 34], [25, 67]],
    [[15, 22], [7, 44]],
    [[158, 35], [30, 158]],
    [[107, 1], [1, 91]]
]
labels = ['KNN', 'SVM', 'Ridge', 'GBR', 'RF', 'SAINT', 'TLTD', 'MfePFN']
truth_labels = ['uniform liquid film', 'elliptical liquid film']
predicted_labels = ['uniform liquid film', 'elliptical liquid film']

# Create figure and subplots
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

# Define a color map where larger numbers are darker
cmap = plt.cm.RdPu  # Red-Purple colormap

# Plot each confusion matrix
for i, ax in enumerate(axes):
    cm = confusion_data[i]
    cm = np.array(cm)

    # Calculate the maximum value in the matrix for normalization
    max_val = np.max(cm)

    # Create a mask for diagonal and off-diagonal elements for different intensities
    mask_diag = np.eye(2, dtype=bool)
    mask_off_diag = ~mask_diag

    # Normalize the matrix for the color map
    cm_normalized = cm / max_val

    # Plot the main heatmap with the custom color map and normalization
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False, ax=ax, annot_kws={"fontsize": 24})
    ax.set_ylabel('Truth', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_title(f'{labels[i]}', fontsize=16, fontweight='bold')

    # Set the x and y ticks to display the new labels
    ax.set_xticklabels(predicted_labels, fontsize=10, fontweight='bold')
    ax.set_yticklabels(truth_labels, fontsize=10, fontweight='bold')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.show()