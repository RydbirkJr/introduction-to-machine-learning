from pylab import *
from scipy.io import loadmat
import neurolab as nl
from sklearn import cross_validation
import scipy.linalg as linalg
from scipy import stats
from import_data import *

# Load data from matlab file
data = Data()

data.normalize_data()

y = data.remove_and_get_y('gdp_per_cap')
y = np.matrix(y).T
attributeNames = data.attribute_names
X = data.X
N, M = data.N, data.M
# C = 2 # Hvad er C?

# Parameters for neural network classifier
min_hidden_units = 1  # number of hidden units
n_hidden_units = 10  # number of hidden units
# n_train = 1  # number of networks trained in each k-fold
learning_goal = 0.5  # stop criterion 1 (train mse to be reached)
max_epochs = 10  # stop criterion 2 (max epochs in training)
show_error_freq = 200  # frequency of training status updates

# K-fold crossvalidation
K = 10  # only five folds to speed up this example
CV = cross_validation.KFold(N, K, shuffle=True)

errors = np.zeros((max_epochs, K))
train_errors = np.zeros((max_epochs, K))
error_hist = np.zeros((max_epochs, K))
bestnet = list()
k = 0
for train_index, test_index in CV:
    print('\nCrossvalidation fold: {0}/{1}'.format(k + 1, K))

    # extract training and test set for current CV fold
    X_train = X[train_index, :]
    y_train = y[train_index, :]
    X_test = X[test_index, :]
    y_test = y[test_index, :]

    best_train_error = 1e100

    ann = nl.net.newff([[-3, 3]] * M, [n_hidden_units, 1], [nl.trans.TanSig(), nl.trans.PureLin()])
    bestnet.append(ann)

    for epoch in range(max_epochs):
        # print('Training network {0}/{1}...'.format(i + 1, n_train))
        # Create randomly initialized network with 2 layers

        # train network
        train_error = ann.train(X_train, y_train, goal=learning_goal, epochs=1, show=show_error_freq)

        # if train_error[-1] < best_train_error:
        #     bestnet[k] = ann
        #     best_train_error = train_error[-1]
        #     error_hist[range(len(train_error)), k] = train_error

        print('Training error for hidden units {0}, k-fold {1}: {2}...'.format(epoch, k, train_error[-1]))
        y_est = ann.sim(X_test)
        errors[epoch][k] = np.square(y_est - y_test).sum().astype(float) / y_test.shape[0]

        train_est = ann.sim(X_train)
        train_errors[epoch][k] = np.square(train_est - y_train).sum().astype(float) / y_train.shape[0]
        print('Training error calculated: {0}'.format(train_errors[epoch][k]))
        print('Test error calculated: {0}'.format(errors[epoch][k]))
    k += 1

# Print the average least squares error
print('Mean-square error: {0}'.format(mean(errors)))

figure()
plot(sum(train_errors, 1) / K)
plot(sum(errors, 1) / K)
legend(['Train error', 'Test error'])
xlabel('Epochs')
ylabel('Mean squared error')
title('Error per epoch', fontweight='bold')
savefig('img/ann_regression_epochs.pdf')
show()

#
# figure();
# subplot(2, 1, 1);
# bar(range(0, K), errors);
# title('Mean-square errors');
# subplot(2, 1, 2);
# plot(error_hist);
# title('Training error as function of BP iterations');
# figure();
# subplot(2, 1, 1);
# plot(y_est);
# plot(y_test.A);
# title('Last CV-fold: est_y vs. test_y');
# subplot(2, 1, 2);
# plot((y_est - y_test).A);
# title('Last CV-fold: prediction error (est_y-test_y)');
# show()
# show()



# # Load Matlab data file and extract variables of interest
# mat_data = loadmat('../Data/wine2.mat')
# X = np.matrix(mat_data['X'])
# y = X[:, 10]  # alcohol contents (target)
# X = X[:, 1:10]  # the rest of features
# N, M = X.shape
# C = 2

# Normalize data
# X = stats.zscore(X);

# Normalize and compute PCA (UNCOMMENT to experiment with PCA preprocessing)
# Y = stats.zscore(X,0);
# U,S,V = linalg.svd(Y,full_matrices=False)
# V = mat(V).T
# Components to be included as features
# k_pca = 3
# X = X*V[:,0:k_pca]
# N, M = X.shape

# Variable for classification error
