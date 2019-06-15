from os import listdir
import numpy as np
import skimage
import scipy.sparse.linalg as sla

# 2a
# load all images in a directory
tmp_images = []
for filename in listdir('yalefaces_cropBackground'):
    tmp_images.append(np.ndarray.flatten(
                    skimage.data.imread('yalefaces_cropBackground/' + filename)))
    images = np.concatenate([image[np.newaxis] for image in tmp_images])

# 2b
# compute the mean face 
num_images = len(images)
sum_faces = np.sum(images, axis=0)
mean_face = 1/num_images * sum_faces

# center the whole data matrix
# X_tilde = X - 1_n * np.transpose(mean_face)
images_centered = np.subtract(images, np.transpose(mean_face))
# print(images,'\n\n\n', mean_face,'\n\n\n', images_centered)

# 2c
# Compute the singular value decomposition X_tilde = U*D*V_transposed for the centered data matrix
# X_tilde = U*D*np.transpose(V)
num_eigenvalues = 3  # TODO: correct? must be between 1 and min(images_centered.shape), k=3 due to u, s, vt?
# u Unitary matrix having left singular vectors as columns.
# s are the singular values
# vt are Unitary matrix having right singular vectors as rows
u, s, vt = sla.svds(images_centered, k=num_eigenvalues)  # TODO: u, s, vt don't contain singular values?
print('u=', u, ', s=', s, ', vt=', vt)

# 2d
# Find the p-dimensional representations Z = X_tilde * V_p, where Vp ∈ R^77760×p 
# contains only the first p columns of V. 
# Assume p = 60. The rows of Z represent each face as a p-dimensional vector, 
# instead of a 77760-dimensional image.
p = 60
