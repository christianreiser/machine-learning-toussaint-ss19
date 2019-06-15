from os import listdir
import numpy as np
import skimage
import scipy.sparse.linalg as sla
from PIL import Image

# 2a
# load all images in a directory
tmp_images = []
for filename in listdir('yalefaces'):
    tmp_images.append(np.ndarray.flatten(
                    skimage.data.imread('yalefaces/' + filename)))
    images = np.concatenate([image[np.newaxis] for image in tmp_images])

# 2b
# compute the mean face 
num_images = images.shape[0]
sum_faces = np.sum(images, axis=0)
mean_face = 1/num_images * sum_faces

# center the whole data matrix
# X_tilde = X - 1_n * np.transpose(mean_face)
images_centered = np.subtract(images, np.transpose(mean_face))
# print(images,'\n\n\n', mean_face,'\n\n\n', images_centered)

# 2c
# Compute the singular value decomposition X_tilde = U*D*V_transposed for the centered data matrix
# X_tilde = U*D*np.transpose(V)
num_eigenvalues = 60  # p value of ex. 2d
# u Unitary matrix having left singular vectors as columns.
# s are the singular values in ascending order (will be flipped)
# vt Unitary matrix having right singular vectors as rows
u, s, vt = sla.svds(images_centered, k=num_eigenvalues)
s = np.flip(s)  # flip to descending order
u = np.flip(u)
vt = np.flip(vt)
# print('\n\n u=', u, ',\n\n s=', s, ',\n\n vt=', vt)

# 2d
# Find the p-dimensional representations Z = X_tilde * V_p, where Vp ∈ R^77760×p 
# contains only the first p columns of V. 
# Assume p = 60. The rows of Z represent each face as a p-dimensional vector, 
# instead of a 77760-dimensional image.
v_p = np.transpose(vt)  # p contains only the first p columns of V
Z = np.matmul(images_centered, v_p)
print('v_p shape', v_p.shape)

# Reconstructed_face = mean_face + Z * v_p
reconstructed_images = np.add(
    np.transpose(mean_face),
    np.matmul(Z, np.transpose(v_p)))

# save and display images
reconstructed_images_tensor = np.reshape(reconstructed_images, [320, 243, num_images])
#print(reconstructed_images_tensor[0].shape)
#img = Image.fromarray(reconstructed_images, 'L')
#img.save('reconstruction.png')
#img.show()

# compute reconstruction error
reconstruction_error = np.sum(np.subtract(images, reconstructed_images)**2)  # TODO norm elementwise?
print('reconstruction_error=', reconstruction_error)
"""reconstruction_error = 0
for image in range(images):
    reconstruction_error = reconstruction_error + abs()
"""