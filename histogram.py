import numpy as np
import matplotlib.pyplot as plt

# Example grayscale image
img = np.array([
    [0, 23, 42],
    [123, 54, 255],
    [255, 65, 128]
], dtype=np.uint8)

# Flattening the image to 1D
flat = img.flatten()

# Plotting histogram
# plt.hist(flat, bins=256, range=(0, 255), color='gray', alpha=0.7)
plt.hist(flat, color='gray', alpha=0.7)
plt.title("Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
# plt.grid(True)
plt.show()
