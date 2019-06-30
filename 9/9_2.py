
################################################################################3
#Exercise 2
#2a)
import numpy as np
import csv
import random
import math


def Ex2a():
    X = []
    with open('mixture.txt', newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            X.append([float(elem) for elem in row[2:]])

    n = np.shape(X)[0]
    K = 3

    Mu = []
    for k in range(K):
        new_item = random.choice(X)
        while new_item in Mu:
            new_item = random.choice(X)
        Mu.append(new_item)

    Sigma = []
    for i in range(K):
        Sigma.append(np.eye(2))

    Sigma = np.asarray(Sigma)
    X = np.asarray(X)
    Mu = np.asarray(Mu)

    Mu_old = np.copy(Mu)
    Sigma_old = np.copy(Sigma)

    # iteration number
    count = 0

    while count < 101:
        Q = np.zeros((n, K))
        for i in range(n):
            for k in range(K):
                Q[i, k] = normal_distr(X[i, :], Mu[k, :], Sigma[k, :, :])
            Q[i, :] = Q[i, :] / np.sum(Q[i, :])

        for k in range(K):
            sum_l = np.sum(Q[:, k])
            sum_u_mu = 0
            sum_u_Sigma = np.zeros((2, 2))
            for i in range(n):
                sum_u_mu += Q[i, k]*X[i, :]

                x_tmp = np.reshape(np.asarray(X[i, :]), (2, 1))  # for matmul
                mu_tmp = np.reshape(np.asarray(Mu[k, :]), (2, 1))
                sum_u_Sigma = np.add(sum_u_Sigma,
                                     Q[i, k]*np.matmul(x_tmp, np.transpose(x_tmp))
                                     - np.matmul(mu_tmp, np.transpose(mu_tmp)))

            # Update the values
            Mu[k, :] = sum_u_mu/sum_l
            Sigma[k, :, :] = sum_u_Sigma/sum_l
        error_Mu = np.linalg.norm(Mu_old - Mu)
        error_Sigma = np.linalg.norm(Sigma_old - Sigma)
        Mu_old = np.copy(Mu)
        Sigma_old = np.copy(Sigma)
        count = count + 1
    print('Error Mu: ', error_Mu)
    print('Error Sigma: ', error_Sigma)


def Ex2b():
    X = []
    with open('mixture.txt', newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            X.append([float(elem) for elem in row[2:]])

    n = np.shape(X)[0]
    K = 3

    Mu = []
    for k in range(K):
        new_item = random.choice(X)
        while new_item in Mu:
            new_item = random.choice(X)
        Mu.append(new_item)

    Sigma = []
    for i in range(K):
        Sigma.append(np.eye(2))

    Sigma = np.asarray(Sigma)
    X = np.asarray(X)
    Mu = np.asarray(Mu)

    Mu_old = np.copy(Mu)
    Sigma_old = np.copy(Sigma)

    # iteration number
    count = 0

    # compute prior Q
    Q = np.zeros((n, K))
    for i in range(n):
        for k in range(K):
            Q[i, k] = normal_distr(X[i, :], Mu[k, :], Sigma[k, :, :])
        Q[i, :] = Q[i, :] / np.sum(Q[i, :])

    while count<101:
        for k in range(K):
            sum_l = np.sum(Q[:, k])
            sum_u_mu = 0
            sum_u_Sigma = np.zeros((2,2))
            for i in range(n):
                sum_u_mu += Q[i,k]*X[i, :]

                x_tmp = np.reshape(np.asarray(X[i,:]), (2,1))#for matmul
                mu_tmp = np.reshape(np.asarray(Mu[k, :]), (2,1))
                sum_u_Sigma = np.add(sum_u_Sigma,
                                     Q[i, k]*np.matmul(x_tmp, np.transpose(x_tmp))
                                     - np.matmul(mu_tmp, np.transpose(mu_tmp)))

            # Update the values
            Mu[k, :] = sum_u_mu/sum_l
            Sigma[k, :, :] = sum_u_Sigma/sum_l
        error_Mu = np.linalg.norm(Mu_old - Mu)
        error_Sigma = np.linalg.norm(Sigma_old - Sigma)
        Mu_old = np.copy(Mu)
        Sigma_old = np.copy(Sigma)
        count = count + 1

        # Compute posterior Q:
        tmp_clusters = np.random.random_integers(0, K, n)
        for i in range(n):
            for k in range(K):
                if tmp_clusters[i] == k:
                    Q[i, k] = 1
                else:
                    Q[i, k] = 0
    print('Error Mu: ', error_Mu)
    print('Error Sigma: ', error_Sigma)


def main():
    Ex2b()


def normal_distr(x, mu, Sigma):
    det = np.linalg.det(Sigma)

    return 1/(2*math.pi*abs(det))**(1/2)*math.exp(
        -1/2*np.matmul(
            np.matmul(
                np.transpose(np.subtract(x, mu)), np.linalg.inv(Sigma)),
            np.subtract(x, mu))
    )


main()
