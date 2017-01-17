# exercise 6.2.1

from pylab import *
from import_data import *
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import cross_validation, tree
from scipy import stats
from import_data import *
from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from linear_regression import *
import neurolab as nl

data = Data()
data.normalize_data()

y = np.matrix(data.remove_and_get_y('gdp_per_cap')).T
X = np.matrix(data.X)

#classNames = data.cont_classes
attributeNames = data.attribute_names

N, M = X.shape

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = cross_validation.KFold(N, K, shuffle=True)
# CV = cross_validation.StratifiedKFold(y.A.ravel(),k=K)

# Initialize variables
Error_ann = np.empty((K, 1))
Error_lin_reg = np.empty((K, 1))
n_tested = 0

k = 0
for train_index, test_index in CV:
    # extract training and test set for current CV fold
    X_train = X[train_index, :]
    y_train = y[train_index, :]
    X_test = X[test_index, :]
    y_test = y[test_index, :]

    ann = nl.net.newff([[-3, 3]] * M, [10, 1], [nl.trans.TanSig(), nl.trans.PureLin()])

    # train network
    train_error = ann.train(X_train, y_train, goal=0.5, epochs=10, show=200)
    y_est = ann.sim(X_test)
    Error_ann[k] = np.square(y_est - y_test).sum().astype(float) / y_test.shape[0]

    # train lin reg
    m = lm.LinearRegression().fit(X_train[:, best_selected_features], y_train)
    # Error_train_fs[k] = np.square(y_train - m.predict(X_train[:, best_selected_features])).sum() / y_train.shape[0]
    Error_lin_reg[k] = np.square(y_test - m.predict(X_test[:, best_selected_features])).sum() / y_test.shape[0]

    k += 1

# Use T-test to check if classifiers are significantly different
[tstatistic, pvalue] = stats.ttest_ind(Error_ann, Error_lin_reg)
if pvalue <= 0.05:
    print('Classifiers are significantly different. (p={0})'.format(pvalue[0]))
else:
    print('Classifiers are not significantly different (p={0})'.format(pvalue[0]))

# Boxplot to compare classifier error distributions
figure()
suptitle('ANN vs. Linear Regression', fontweight='bold')
boxplot(np.bmat('Error_ann, Error_lin_reg'))
xlabel('ANN vs. Linear Regression')
ylabel('Mean squared error')
savefig('img/t_test_regression.pdf')
show()
