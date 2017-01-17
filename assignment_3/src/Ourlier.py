# exercise 11.2.3
from pylab import *
from sklearn.neighbors import NearestNeighbors
from import_data import *
from sklearn import cross_validation

#------------------------

data = Data()

data.normalize_data()
X = np.matrix(data.X)
N, M = data.N, data.M

CV = cross_validation.LeaveOneOut(data.N)
#------------------------



for train_index, test_index in CV:
    print('Crossvalidation fold: {0}/{1}'.format(i + 1, data.N))

    # extract training and test set for current CV fold
    X_train = data.X[train_index, :]
    X_test = data.X[test_index, :]
    # Number of neighbors
    K = 137

    # Find the k nearest neighbors
    knn = NearestNeighbors(n_neighbors=K).fit(X)
    D, i = knn.kneighbors(X)

    # Compute the density
    #D, i = knclassifier.kneighbors(np.matrix(xe).T)
    knn_density = 1./(D.sum(axis=1)/K)

    # Compute the average relative density
    DX, iX = knn.kneighbors(X)
    knn_densityX = 1./(DX[:,1:].sum(axis=1)/K)
    knn_avg_rel_density = knn_density/(knn_densityX[i[:,1:]].sum(axis=1)/K)


# Plot KNN density
figure()
subplot(2,1,1)
hist(X,X)
title('Data histogram')
subplot(2,1,2)
plot(X, knn_density)
title('KNN density')
# Plot KNN average relative density
figure()
subplot(2,1,1)
hist(X,x)
title('Data histogram')
subplot(2,1,2)
plot(xe, knn_avg_rel_density)
title('KNN average relative density')

show()
