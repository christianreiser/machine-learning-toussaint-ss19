from os import listdir
import numpy as np
import skimage
from PIL import Image
# import KMeans

# 1adistance_pixel_sum_nearest_cluster
# K = 4, random initializations of the centers, Repeat k-means clustering 10 times.
# For each run, report on the clustering error.
# pick the best clustering. Display the center faces
images_tmp = []
for filename in listdir('yalefaces_cropBackground'):
    images_tmp.append(skimage.data.imread('yalefaces_cropBackground/' + filename))
    D = np.reshape(images_tmp, (len(images_tmp), -1))
print('D.shape', D.shape)

num_imgs = D.shape[0]
len_flat_img = D.shape[1]

# initialize
num_iterations = 9
num_clusters = 15  # number of clusters
runs = 10
for num_clusters in range(4, 10, num_imgs):  # test for different k
    print('num_clusters', num_clusters)
    for run in range(runs):
        print('run:', run)
        cluster_mean = np.random.randint(low=0, high=255, size=[num_clusters, len_flat_img])  # ini cluster mean randomly
        distances = np.zeros([num_imgs, len_flat_img, num_clusters])  # ini centroid data distance tensor
        distances_pixel_sum = np.zeros([len_flat_img, num_clusters])
        # distances in first column and cluster index in second (c in sheet)
        distance_pixel_sum_nearest_cluster = np.zeros([len_flat_img, 2])
        error = 999999999

        for iteration in range(num_iterations):
            print('iteration:', iteration)
            # assign images to clusters
            for cluster in range(num_clusters):  # get distances
                distances[:, :, cluster] = abs(np.subtract(D, cluster_mean[cluster]))  # subtract cluster mean from evey pixel
                # distances[:, :, cluster] = np.power(distances[:, :, cluster], 2)  # TODO: power on, careful for mean distance
                for image in range(num_imgs):
                    distances_pixel_sum[image, cluster] = np.sum(distances[image, :, cluster])  # sum pixel distances
            for image in range(num_imgs):  # get distance nearest cluster and assign cluster to images
                nearest_cluster_index = np.argmin(distances_pixel_sum[image, :])  # nearest cluster index
                distance_pixel_sum_nearest_cluster[image, 1] = nearest_cluster_index  # store nearest cluster index
                distance_pixel_sum_nearest_cluster[image, 0] = distances_pixel_sum[image, nearest_cluster_index]  # nearest distance
            error_new = np.sum(distance_pixel_sum_nearest_cluster[:, 0])

            print('error', error_new)
            if error_new == error:
                break
            else:
                error = error_new



            # cluster images
            image_cluster = np.zeros([num_clusters, num_imgs, len_flat_img])
            num_images_in_cluster = np.zeros(num_clusters)  # ini number of images in cluster counter
            for image in range(num_imgs):
                for cluster in range(num_clusters):
                    if distance_pixel_sum_nearest_cluster[image, 1] == cluster:
                        image_cluster[cluster, image, :] = D[image, :]
                        num_images_in_cluster[cluster] += 1
            print('num_images_in_cluster', num_images_in_cluster)

            # relocate cluster cluster_mean)
            num_reinitializations = 0
            for cluster in range(num_clusters):
                for pixel in range(len_flat_img):
                    # average pixel of image cluster
                    mean_pixel = 1/num_images_in_cluster[cluster] * np.sum(image_cluster[cluster, :, pixel])
                    # print('mean_pixel', mean_pixel)
                    if mean_pixel > 0.1:  # many mean pixels are NaN, because cluster has no assigned image
                        cluster_mean[cluster, pixel] = mean_pixel
                        # print('high mean pixel value', mean_pixel)
                    else:
                        # print('low mean pixel value', mean_pixel)
                        # reinitialize cluster mean pixel randomly
                        cluster_mean[cluster, pixel] = np.random.randint(low=0, high=255)
                        num_reinitializations += 1
            print('num reinitializations', num_reinitializations)

        # show mean faces
        for cluster in range(num_clusters):
            img_reshaped = np.reshape(cluster_mean[cluster, :], [243, 160])
            img_reshaped = img_reshaped.astype('uint8')  # change to uint8 otherwise img.show doesnt work
            img = Image.fromarray(img_reshaped, 'L')
            img.save('mean_faces/k_'+str(num_clusters)+'run'+str(run)+'_mean_face_cluster'+str(cluster)+'.png')
    error_over_k = error_over_k

    img.show()

print('end')
