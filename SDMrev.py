import numpy as np
import matplotlib.pyplot as plt


img = np.array([
  [120, 23,  42],
  [123,54,255],
  [255, 65, 128]
], dtype=np.uint8)


# plt.imshow(img, cmap='gray', vmin=0, vmax=255)
# plt.title("Tiny Grayscale Image")
# plt.colorbar(label="intensity")
# plt.show()


# Negative Transformation
negative = 255 - img
# plt.imshow(negative, cmap='gray', vmin=0, vmax=255)
# plt.title("Negative Image")
# plt.colorbar(label="intensity")
# plt.show()


# Contrast Stretching
r_min, r_max = img.min(), img.max()
r_min1, r_max1 = negative.min(), negative.max()
constrast_stretched = ((img - r_min) / (r_max - r_min) * 255).astype(np.uint8)
constrast_stretched_negative = ((negative - r_min1) / (r_max1 - r_min1) * 255).astype(np.uint8)


# Log Transformation
img_float = img.astype(np.float32)
c_log = 255 / np.log1p(np.max(img_float))
log_trans = c_log * np.log1p(img_float)
log_trans = np.clip(log_trans, 0, 255).astype(np.uint8)



# Gammma Transformation(γ < 1 for brighter image)
gamma = 1.5
gamma_corrected = np.power(img / 255.0, gamma)
gamma_corrected = np.uint8(gamma_corrected * 255)


images = [img, negative, constrast_stretched, constrast_stretched_negative, log_trans, gamma_corrected]
titles = ["Original", "Negative", "Contrast_Stretched", "Negative Contrast Image", "log_transformation", "Gamma Transformation"]

plt.figure(figsize=(12,3))
n = len(images)
for i in range(n):
  plt.subplot(1, n, i + 1)
  plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
  plt.title(titles[i])
  plt.axis('off')

plt.tight_layout()
plt.show()