import numpy as np
import matplotlib.pyplot as plt

# Generate a random NumPy array of size 48x2
data = np.random.rand(48, 2)

# Create a figure and axes
fig, ax = plt.subplots()

# Create a heat map using the imshow function
heatmap = ax.imshow(data, cmap='hot')

# Add a colorbar to the heat map
cbar = plt.colorbar(heatmap)

# Show the plot
plt.show()