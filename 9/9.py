# TODO: my has same dim as image


from os import listdir
import numpy as np
import skimage
# import KMeans
# from PIL import Image

# 1adistance_sum_nearest_cluster
# K = 4, random initializations of the centers, Repeat k-means clustering 10 times.
# For each run, report on the clustering error.
# pick the best clustering. Display the center faces
images_tmp = []
for filename in listdir('yalefaces_cropBackground'):
    images_tmp.append(skimage.data.imread('yalefaces_cropBackground/' + filename))
    D = np.reshape(images_tmp, (len(images_tmp), -1))
print('D.shape', D.shape)
print('first image', D[0, :])
num_imgs = D.shape[0]
len_flat_img = D.shape[1]

# initialize
num_iterations = 2
num_clusters = 4  # number of clusters
cluster_mean = np.random.randint(low=0, high=255, size=[num_clusters, len_flat_img])  # ini cluster mean randomly
print('initial cluster mean', cluster_mean)
# data_class = np.zeros(num_imgs)distance_sum_nearest_cluster
distances = np.zeros([num_imgs, len_flat_img, num_clusters])  # ini centroid data distance tensor
distances_sum = np.zeros([len_flat_img, num_clusters])
# distances in first column and cluster index in second (c in sheet)
distance_sum_nearest_cluster = np.zeros([len_flat_img, 2])

for iteration in range(num_iterations):
    # assign images to clusters
    for cluster in range(num_clusters):  # get distances
        distances[:, :, cluster] = np.subtract(D, cluster_mean[cluster])  # subtract cluster mean from evey pixel
        # distances[:, :, cluster] = np.power(distances[:, :, cluster], 2)  # TODO: power on, careful for mean distance
        for image in range(num_imgs):
            distances_sum[image, cluster] = np.sum(distances[image, :, cluster])  # sum pixel distances
    for image in range(num_imgs):  # get distance nearest cluster and assign cluster to images
        nearest_cluster_index = np.argmin(distances_sum[image, :])  # nearest cluster index
        distance_sum_nearest_cluster[image, 1] = nearest_cluster_index  # store nearest cluster index
        distance_sum_nearest_cluster[image, 0] = distances_sum[image, nearest_cluster_index]  # nearest distance
    print('distance_sum_nearest_cluster[:, 0]', np.sum(distance_sum_nearest_cluster[:, 0]))

    # cluster images
    image_cluster = np.zeros([num_clusters, num_imgs, len_flat_img])
    num_images_in_cluster = np.zeros(num_clusters)  # ini number of images in cluster counter
    for image in range(num_imgs):
        if distance_sum_nearest_cluster[image, 1] == 0:
            image_cluster[0, image, :] = D[image, :]
            num_images_in_cluster[0] += 1
        elif distance_sum_nearest_cluster[image, 1] == 1:
            image_cluster[1, image] = D[image, :]
            num_images_in_cluster[1] += 1
        elif distance_sum_nearest_cluster[image, 1] == 2:
            image_cluster[2, image] = D[image, :]
            num_images_in_cluster[2] += 1
        elif distance_sum_nearest_cluster[image, 1] == 3:
            image_cluster[3, image] = D[image, :]
            num_images_in_cluster[3] += 1
        else:
            print('WARNING: falsely assigned!')
    print('num_images_in_cluster', num_images_in_cluster)

    # relocate cluster cluster_mean)
    num_reinitializations = 0
    for cluster in range(num_clusters):
        for pixel in range(len_flat_img):
            # average pixel of image cluster
            mean_pixel = 1/num_images_in_cluster[cluster] * np.sum(image_cluster[cluster, :, pixel])
            # print('mean_pixel', mean_pixel)
            if mean_pixel > 0.5:  #
                cluster_mean[cluster, pixel] = mean_pixel
                # print('high mean pixel value', mean_pixel)
            else:
                # print('low mean pixel value', mean_pixel)
                # reinitialize cluster mean pixel randomly
                cluster_mean[cluster, pixel] = np.random.randint(low=0, high=255)
                num_reinitializations += 1
    print('num reinitializations', num_reinitializations)

print('assigned clusters of images:', distance_sum_nearest_cluster[:, 1])
print('cluster mean=', cluster_mean)
"""
n_init=10 to run 10 times with different centroid seeds.
        The final results will be the best output of n_init consecutive runs in terms of inertia.
verbose=1 to print inertia, which is thdistance_sum_nearest_clustere Sum of squared distances of samples to their closest cluster center.
"""
# kmeans = KMeans(n_clusters=9, random_state=None, verbose=1, n_init=10).fit(X)
# print('kmeans:', kmeans)
# print('kmeans.labels_', kmeans.labels_)
