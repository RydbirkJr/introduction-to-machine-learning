# exercise 11.1.1
from pylab import *
from scipy.io import loadmat
from toolbox_02450 import clusterplot
from sklearn.mixture import GMM
# Load Matlab data file and extract variables of interest
from import_data import Data

data = Data()

data.normalize_data()
classNames = data.gov_classes

data.remove_one_out_of_k(classNames)
y = np.matrix(data.y_gov)
attributeNames = data.attribute_names
X = data.get_pca()[:, 0:2]
N, M = X.shape

C = len(classNames)

# Number of clusters

K = 3
cov_type = 'diag'
# type of covariance, you can try out 'diag' as well
reps = 10
# number of fits with different initalizations, best result will be kept
# Fit Gaussian mixture model
gmm = GMM(n_components=K, covariance_type=cov_type, n_init=reps, params='wmc').fit(X)
cls = gmm.predict(X)
# extract cluster labels
cds = gmm.means_
# extract cluster centroids (means of gaussians)
covs = gmm.covars_

# extract cluster shapes (covariances of gaussians)
if cov_type == 'diag':
    new_covs = np.zeros([K,M,M])
    count = 0
    for elem in covs:
        temp_m = np.zeros([M,M])
        for i in range(len(elem)):
            temp_m[i][i] = elem[i]
            new_covs[count] = temp_m
        count += 1
        covs = new_covs
# Plot results:
figure(figsize=(14,9))
print cls.shape
print cds.shape
print y.shape
print covs.shape
clusterplot(X, clusterid=cls, centroids=cds, y=y, covars=covs, classes=classNames)
title('Government Type', fontweight='bold')
ylabel('PC2')
xlabel('PC1')
savefig('img/clusterplot.pdf')
show()
