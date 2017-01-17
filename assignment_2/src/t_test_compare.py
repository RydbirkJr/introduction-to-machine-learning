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

def conv2DS(Xv, yv=None):
    if yv == None:
        yv = np.asmatrix(np.ones((Xv.shape[0], 1)))
        for j in range(len(classNames)): yv[j] = j

    C = len(unique(yv.flatten().tolist()[0]))
    DS = ClassificationDataSet(M, 1, nb_classes=C)
    for i in range(Xv.shape[0]): DS.appendLinked(Xv[i, :].tolist()[0], [yv[i].A[0][0]])
    DS._convertToOneOfMany()
    return DS

def neval(xval):
    return argmax(fnn.activateOnDataset(conv2DS(np.asmatrix(xval))), 1)

data = Data()
data.normalize_data()
X = np.matrix(data.X)
y = np.matrix(data.y_cont).T
data.remove_one_out_of_k(data.cont_classes)
classNames = data.cont_classes
attributeNames = data.attribute_names

N, M = X.shape

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 4
CV = cross_validation.KFold(N, K, shuffle=True)
# CV = cross_validation.StratifiedKFold(y.A.ravel(),k=K)

# Initialize variables
Error_ann = np.empty((K, 1))
Error_dectree = np.empty((K, 1))
n_tested = 0

k = 0
for train_index, test_index in CV:
    # extract training and test set for current CV fold
    X_train = X[train_index, :]
    y_train = y[train_index, :]
    X_test = X[test_index, :]
    y_test = y[test_index, :]

    # Fit and evaluate ANN
    DS_train = conv2DS(X_train, y_train)
    DS_test = conv2DS(X_test, y_test)
    fnn = buildNetwork(DS_train.indim, 16, DS_train.outdim, outclass=SoftmaxLayer, bias=True)
    trainer = BackpropTrainer(fnn, dataset=DS_train, momentum=0.1, verbose=False, weightdecay=0.01)
    for tmp in range(30): trainer.trainEpochs(1)
    ote_test = fnn.activateOnDataset(DS_test)
    ErrorRate_test = (np.argmax(ote_test, 1) != y_test.T).mean(dtype=float) * 100
    Error_ann[k] = ErrorRate_test

    # Fit and evaluate Decision Tree classifier
    # The optimal max depth was found to be 5.
    model2 = tree.DecisionTreeClassifier(criterion="gini", max_depth=3)
    model2 = model2.fit(X_train.A, y_train.A.ravel())
    y_dectree = np.mat(model2.predict(X_test)).T
    Error_dectree[k] = 100 * (y_dectree != y_test).sum().astype(float) / len(y_test)

    k += 1

# Use T-test to check if classifiers are significantly different
[tstatistic, pvalue] = stats.ttest_ind(Error_ann, Error_dectree)
if pvalue <= 0.05:
    print('Classifiers are significantly different. (p={0})'.format(pvalue[0]))
else:
    print('Classifiers are not significantly different (p={0})'.format(pvalue[0]))

# Boxplot to compare classifier error distributions
figure()
boxplot(np.bmat('Error_ann, Error_dectree'))
xlabel('ANN  vs.   Decision Tree')
ylabel('Cross-validation error [%]')
savefig('img/t_test_ann_tree.pdf')
show()