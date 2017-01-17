# exercise 11.1.5
from pylab import *
from scipy.io import loadmat
from import_data import *
from sklearn.mixture import GMM
from toolbox_02450 import clusterplot
from sklearn import cross_validation

# Load Matlab data file and extract variables of interest
data = Data()

data.normalize_data()
classNames = data.gov_classes

data.remove_one_out_of_k(classNames)
y = np.matrix(data.y_gov).T
attributeNames = data.attribute_names
X = data.X #data.get_pca()[:, :2]
N, M = data.N, data.M

C = len(classNames)

# Range of K's to try
KRange = range(1, 15)
T = len(KRange)

covar_type = 'diag'  # you can try out 'diag' as well
reps = 10  # number of fits with different initalizations, best result will be kept

# Allocate variables
BIC = np.zeros((T, 1))
AIC = np.zeros((T, 1))
CVE = np.zeros((T, 1))

# K-fold crossvalidation
folds = 10
CV = cross_validation.KFold(N, folds, shuffle=True)

for t, K in enumerate(KRange):
    print('Fitting model for K={0}\n'.format(K))

    # Fit Gaussian mixture model
    gmm = GMM(n_components=K, covariance_type=covar_type, n_init=reps, params='wmc').fit(X)

    # Get BIC and AIC
    BIC[t, 0] = gmm.bic(X)
    AIC[t, 0] = gmm.aic(X)

    for train_index, test_index in CV:
        # extract training and test set for current CV fold
        X_train = X[train_index]
        X_test = X[test_index]

        # Fit Gaussian mixture model to X_train
        gmm = GMM(n_components=K, covariance_type=covar_type, n_init=reps, params='wmc').fit(X_train)

        # compute negative log likelihood of X_test
        CVE[t] += -gmm.score(X_test).sum()

# Plot results
figure(1)
hold(True)
plot(KRange, BIC, color='b')
plot(KRange, AIC, color='y')
plot(KRange, CVE, color='purple')
legend(['BIC', 'AIC', 'Crossvalidation'])
xlabel('K')
ylabel('-log likelihood')
savefig('img/gmm_gov.pdf')
title('GMM')
show()