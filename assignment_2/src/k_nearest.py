from import_data import *
from pylab import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation


data = Data()

L = 40
CV = cross_validation.LeaveOneOut(data.N)
errors = np.zeros((data.N, L))
i = 0

data.normalize_data()
data.remove_one_out_of_k(data.cont_classes)
data.y_cont = np.matrix(data.y_cont).T

# PCA by computing SVD of Y
#U, S, V = linalg.svd(data.X, full_matrices=False)
#V = mat(V).T


# Project the centered data onto principal component space
#Z = data.X * V[:,0:2]

#print V[:,0:3]

C = len(data.cont_classes)

for train_index, test_index in CV:
    print('Crossvalidation fold: {0}/{1}'.format(i + 1, data.N))

    # extract training and test set for current CV fold
    X_train = data.X[train_index, :]
    y_train = data.y_cont[train_index, :]
    X_test = data.X[test_index, :]
    y_test = data.y_cont[test_index, :]

    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    for l in range(1, L + 1):
        knclassifier = KNeighborsClassifier(n_neighbors=l);
        knclassifier.fit(X_train, ravel(y_train));
        y_est = knclassifier.predict(X_test);
        errors[i, l - 1] = np.sum(y_est[0] != y_test[0, 0])

    i += 1





# Plot the training data points (color-coded) and test data points.
figure(1);
hold(True);
styles = ['.b', '.r', '.g', '.y', '.w', '.k']
for c in range(C):
    class_mask = (y_train==c).A.ravel()
    plot(X_train[class_mask,0], X_train[class_mask,1], styles[c])


# Plot the classfication results
styles = ['ob', 'or', 'og', 'oy', 'ow', 'ok']
for c in range(C):
    class_mask = (y_est==c)
    plot(X_test[class_mask,0], X_test[class_mask,1], styles[c], markersize=10)
    plot(X_test[class_mask,0], X_test[class_mask,1], 'kx', markersize=8)
title('Synthetic data classification - KNN');
savefig('img/KNN_scatter_plot.pdf')
# Plot the classification error rate
figure()
plot(100 * sum(errors, 0) / data.N)
xlabel('Number of neighbors')
ylabel('Classification error rate (%)')
savefig('img/Error_rate_KNN.pdf')
show()