import numpy as np


def featureNormalize(X):
    # FEATURENORMALIZE Normalizes the features in X
    #   FEATURENORMALIZE(X) returns a normalized version of X where
    #   the mean value of each feature is 0 and the standard deviation
    #   is 1. This is often a good preprocessing step to do when
    #   working with learning algorithms.

    # You need to set these values correctly
    X_norm = np.transpose(X)
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))

    # ====================== YOUR CODE HERE ======================
    # Instructions: First, for each feature dimension, compute the mean
    #               of the feature and subtract it from the dataset,
    #               storing the mean value in mu. Next, compute the
    #               standard deviation of each feature and divide
    #               each feature by it's standard deviation, storing
    #               the standard deviation in sigma.
    #
    #               Note that X is a matrix where each column is a
    #               feature and each row is an example. You need
    #               to perform the normalization separately for
    #               each feature. Column i of X is :
    #  				X[:, i]
    #
    # Hint: You might find the 'np.mean()' and 'np.std()' functions useful.
    #

    # ============================================================
    for n in range(len(X[0])):
        mu[0, n] = np.mean(X[:, n])
        sigma[0, n] = np.std(X[:, n])
        X_norm[n] = np.subtract(X_norm[n], np.full((1, (len(X_norm[n]))), mu[0, n]))
        X_norm[n] = np.multiply(X_norm[n], np.full((1, (len(X_norm[n]))), (1 / sigma[0, n])))
    X_norm = np.transpose(X_norm)

    return [X_norm, mu, sigma]
