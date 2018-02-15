import numpy as np


def sample_zero_mean(x):
    row_mean = np.mean(x, axis=1).reshape(np.shape(x)[0], 1)
    x_scaled = x - np.dot(row_mean, np.ones((1, np.shape(x)[1])))
    return x_scaled


def gcn(x, scale=55., bias=0.01):
    gcn_coefficient = scale / np.sqrt(np.var(x, axis=1) + bias)
    return (gcn_coefficient * x.T).T


def feature_zero_mean(x, xtest):
    mean_fea = np.mean(x, axis=0).reshape(1, np.shape(x)[1])
    x_scaled = x - np.dot(np.ones((np.shape(x)[0], 1)), mean_fea)
    xtest_scaled = xtest - np.dot(np.ones((np.shape(xtest)[0], 1)), mean_fea)
    return x_scaled, xtest_scaled


def zca(x, xtest, bias=0.1):
    sigma = np.dot(x.T, x) / np.shape(x)[0]
    u, s, _ = np.linalg.svd(sigma)
    principal_comps = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + bias))), u.T)
    return np.dot(x, principal_comps), np.dot(xtest, principal_components)


def cifar_10_preprocess(x, xtest, image_size=32):
    x_scaled = gcn(sample_zero_mean(x))
    xtest_scaled = gcn(sample_zero_mean(xtest))
    x_zca, xtest_zca = zca(feature_zero_mean(x_scaled, xtest_scaled)[0],
                           feature_zero_mean(x_scaled, xtest_scaled)[1])
    x_reshape = np.reshape(x_zca, (np.shape(x_zca)[0], 3,
                                   image_size, image_size))
    xtest_reshape = np.reshape(xtest_zca, (np.shape(xtest_zca)[0],
                                           3, image_size, image_size))
    return x_reshape, xtest_reshape
