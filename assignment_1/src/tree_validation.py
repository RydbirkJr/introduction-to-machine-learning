# exercise 6.1.2

from pylab import *
from scipy.io import loadmat
from sklearn import cross_validation, tree
from import_data import Data

# Load Matlab data file and extract variables of interest

data = Data()
print data.M
data.remove_one_out_of_k(data.cont_classes)
print data.M

#mat_data = loadmat('../Data/wine2.mat')
X = np.matrix(data.X)
y = np.matrix(data.y_cont).T #np.matrix(data.y_gov) #np.matrix(mat_data['y'], dtype=int)
attributeNames = data.attribute_names #[name[0] for name in mat_data['attributeNames'][0]]
classNames = data.cont_classes #[name[0][0] for name in mat_data['classNames']]
N = data.N
M = data.M
C = data.C_cont

# for class_name in classNames:
#     index = attributeNames.index(class_name)
#     attributeNames.pop(index)
#     X = np.delete(X, [index], 1)
#
# index = attributeNames.index('gov_type')
# attributeNames.pop(index)
# X = np.delete(X, [index], 1)
#
# index = attributeNames.index('continent')
# attributeNames.pop(index)
# X = np.delete(X, [index], 1)

print attributeNames
print N, M
print y
print 'y: ', len(y[:,:].T)

#N, M = X.shape
#C = len(classNames)

# Tree complexity parameter - constraint on maximum depth
tc = np.arange(2, 21, 1)

# K-fold crossvalidation
K = 10
CV = cross_validation.KFold(N, K, shuffle=True)

# Initialize variable
Error_train = np.empty((len(tc), K))
Error_test = np.empty((len(tc), K))

k = 0
for train_index, test_index in CV:
    print('Computing CV fold: {0}/{1}..'.format(k + 1, K))
    #print train_index, test_index

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index, :].A, y[train_index, :].A
    X_test, y_test = X[test_index, :].A, y[test_index, :].A

    for i, t in enumerate(tc):
        # Fit decision tree classifier, Gini split criterion, different pruning levels
        dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
        dtc = dtc.fit(X_train, y_train.ravel())
        y_est_test = dtc.predict(X_test)
        y_est_train = dtc.predict(X_train)
        # Evaluate misclassification rate over train/test data (in this CV fold)
        misclass_rate_test = sum(np.abs(np.mat(y_est_test).T - y_test)) / float(len(y_est_test))
        misclass_rate_train = sum(np.abs(np.mat(y_est_train).T - y_train)) / float(len(y_est_train))
        Error_test[i, k], Error_train[i, k] = misclass_rate_test, misclass_rate_train
    k += 1

f = figure()
f.hold(True)
boxplot(Error_test.T)
xlabel('Model complexity (max tree depth)')
ylabel('Test error across CV folds, K={0})'.format(K))

f = figure();
f.hold(True)
plot(tc, Error_train.mean(1))
plot(tc, Error_test.mean(1))
xlabel('Model complexity (max tree depth)')
ylabel('Error (misclassification rate, CV K={0})'.format(K))
legend(['Error_train', 'Error_test'])

show()
