# exercise 6.2.1


from pylab import *
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import cross_validation
from toolbox_02450 import feature_selector_lr, bmplot
from import_data import *
import matplotlib.ticker as ticker

# Load data from matlab file
data = Data()



# gdp_cap = data.X[:, data.attribute_names.index('gdp_per_cap')]
# newX = np.zeros((data.N, data.M + 1))
# newX[:, :-1] = data.X
# newX[:, -1] = np.log(gdp_cap)
#
# data.X = newX
#data.attribute_names.append('gdp_cap_log')

data.normalize_data()

#y = data.remove_and_get_y('unemployment_rate')
y = data.remove_and_get_y('gdp_per_cap')

X = data.X

N, M = X.shape

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
attributeNames = [u'Offset'] + data.attribute_names
M = M + 1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = cross_validation.KFold(N, K, shuffle=True)

# Initialize variables
Features = np.zeros((M, K))
Error_train = np.empty((K, 1))
Error_test = np.empty((K, 1))
Error_train_fs = np.empty((K, 1))
Error_test_fs = np.empty((K, 1))
Error_mean = np.empty((K, 1))
Error_train_nofeatures = np.empty((K, 1))
Error_test_nofeatures = np.empty((K, 1))

k = 0
best_error = 1e32
best_k = 0
best_loss = []
best_rec = []
best_selected_features = []
best_params = []
for train_index, test_index in CV:
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10

    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train - y_train.mean()).sum() / y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test - y_test.mean()).sum() / y_test.shape[0]

    # Compute squared error with all features selected (no feature selection)
    m = lm.LinearRegression().fit(X_train, y_train)
    Error_train[k] = np.square(y_train - m.predict(X_train)).sum() / y_train.shape[0]
    Error_test[k] = np.square(y_test - m.predict(X_test)).sum() / y_test.shape[0]

    # Compute squared error with feature subset selection
    selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation)
    Features[selected_features, k] = 1
    # .. alternatively you could use module sklearn.feature_selection
    print selected_features
    m = lm.LinearRegression().fit(X_train[:, selected_features], y_train)
    Error_train_fs[k] = np.square(y_train - m.predict(X_train[:, selected_features])).sum() / y_train.shape[0]
    Error_test_fs[k] = np.square(y_test - m.predict(X_test[:, selected_features])).sum() / y_test.shape[0]
    Error_mean[k] = np.square(y_test - y_test.mean()).sum() / y_test.shape[0]


#    print('Test error for ', k, ': ', Error_test_fs[k])

    if Error_test_fs[k] < best_error:
        best_error = Error_test_fs[k]
        best_k = k
        best_loss = loss_record
        best_rec = features_record
        best_selected_features = selected_features
        best_params = m.coef_

    print('Cross validation fold {0}/{1}'.format(k + 1, K))
    # print('Train indices: {0}'.format(train_index))
    # print('Test indices: {0}'.format(test_index))
    print('Features no: {0}\n'.format(selected_features.size))

    k += 1


def plot_feature_selection(loss, attribute_names, feature_records):
    figure()
    suptitle('Forward Selection of Features', fontweight='bold')

    subplot(1, 3, 1)
    plot(range(1, len(loss)), loss[1:])
    locs, labels = xticks()
    xticks(locs, map(lambda x: "%1.0f" % x, locs))
    xlabel('Iteration')
    ylabel('Squared Error (Crossvalidation)')

    subplot(1, 3, 3)
    bmplot(attribute_names, range(1, feature_records.shape[1]), -feature_records[:, 1:])
    clim(-1.5, 0)
    xlabel('Iteration')
    savefig('img/best_forward_selection.pdf')

plot_feature_selection(best_loss, attributeNames, best_rec)

print('Weights of best model')
print(best_params)


# Display results
# print('\n')
# print('Linear regression without feature selection:\n')
# print('- Training error: {0}'.format(Error_train.mean()))
# print('- Test error:     {0}'.format(Error_test.mean()))
# print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum() - Error_train.sum()) / Error_train_nofeatures.sum()))
# print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum() - Error_test.sum()) / Error_test_nofeatures.sum()))
# print('Linear regression with feature selection:\n')
print('- Training error: {0}'.format(Error_train_fs.mean()))
print('- Test error:     {0}'.format(Error_test_fs.mean()))
print('- Mean error:     {0}'.format(Error_mean.mean()))
# print(
# '- R^2 train:     {0}'.format((Error_train_nofeatures.sum() - Error_train_fs.sum()) / Error_train_nofeatures.sum()))
# print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum() - Error_test_fs.sum()) / Error_test_nofeatures.sum()))

figure(k)
#subplot(1, 3, 2)
suptitle('Selected Features per Crossvalidation Fold', fontweight='bold')
bmplot(attributeNames, range(1, Features.shape[1] + 1), -Features)
clim(-1.5, 0)
xlabel('Crossvalidation Fold')
#ylabel('Attribute')
savefig('img/linear_k_fold.pdf')

# Inspect selected feature coefficients effect on the entire dataset and
# plot the fitted model residual error as function of each attribute to
# inspect for systematic structure in the residual
for f in range(best_k, best_k+1):
    # f = 2  # cross-validation fold to inspect
    ff = Features[:, f].nonzero()[0]
    m = lm.LinearRegression().fit(X[:, ff], y)

    y_est = m.predict(X[:, ff])
    residual = y - y_est

    figure(k + f)
    suptitle('Residual Error for Selected Features', fontweight='bold')
    for i in range(0, len(ff)):
        ax = subplot(2, ceil(len(ff) / 2.0), i + 1)
        plot(X[:, ff[i]], residual, '.')
        xlabel(attributeNames[ff[i]])
        if i == 0 or i == (len(ff)+1) / 2:
            ylabel('Residual Error')

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
#        locs, labels = xticks()
 #       xticks(locs, map(lambda x: "%1.0f" % x, locs))

        #xticks(np.arange(min(X[:, ff[i]]), max(X[:, ff[i]]) + 1, 1.0))

        savefig('img/residual_error.pdf')

show()
