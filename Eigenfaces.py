import cv2
import numpy as np
import os
from tqdm import tqdm  # Progress bar

# Path to images
IMAGE_FOLDER = "images/"
IMG_SIZE = (100, 100)  # Resize all images to this size

# Store images as vectors
faces = []
image_filenames = []

# Read all images in the folder
for i in tqdm(range(1, 6986)):  # Adjust range if necessary
    filename = f"{IMAGE_FOLDER}{i:06d}.jpg"  # Format 000001.jpg, 000002.jpg, etc.
    if os.path.exists(filename):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        img = cv2.resize(img, IMG_SIZE)  # Resize to uniform size
        faces.append(img.flatten())  # Flatten to 1D vector
        image_filenames.append(filename)

# Convert list to NumPy array
faces = np.array(faces)
print(f"Loaded {len(faces)} images.")

# Create the data matrix where each row is a flattened image
data_matrix = np.array(faces, dtype=np.float32)
print(f"Data matrix shape: {data_matrix.shape}")  # (N, 10000)


# Compute the mean face vector
mean_face = np.mean(data_matrix, axis=0)

# Subtract the mean face from all images
normalized_faces = data_matrix - mean_face

# Reshape and display the mean face
mean_face_img = mean_face.reshape(IMG_SIZE)

import matplotlib.pyplot as plt

plt.imshow(mean_face_img, cmap='gray')
plt.title("Mean Face")
plt.axis("off")
plt.show()

from sklearn.decomposition import PCA

num_components = 25  # Keep 50 principal components (Eigenfaces)
pca = PCA(n_components=num_components, whiten=True)

# Fit PCA on normalized data
faces_pca = pca.fit_transform(normalized_faces)

print(f"PCA reduced dimensions from {data_matrix.shape[1]} to {num_components}.")

fig, axes = plt.subplots(3, 5, figsize=(12, 6))  # Show first 10 Eigenfaces

for i, ax in enumerate(axes.flat):
    eigenface = pca.components_[i].reshape(IMG_SIZE)
    ax.imshow(eigenface, cmap='gray')
    ax.set_title(f"Eigenface {i+1}")
    ax.axis("off")

plt.show()
