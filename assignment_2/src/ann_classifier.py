# exercise 8.3.3 Fit neural network classifiers using softmax output weighting
from pylab import *
from scipy.io import loadmat
from sklearn import cross_validation
from toolbox_02450 import dbplotf
from import_data import *
from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer


# %% convert to ClassificationDataSet format.
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

N, M = X.shape
C = len(classNames)

K = 5
NHiddenUnits = 20
CV = cross_validation.KFold(N, K, shuffle=True)
Error_train = np.empty((NHiddenUnits, K))
Error_test = np.empty((NHiddenUnits, K))
k = 0
for train_index, test_index in CV:

    X_train, y_train = X[train_index, :], y[train_index, :]
    X_test, y_test = X[test_index, :], y[test_index, :]

    DS_train = conv2DS(X_train, y_train)
    DS_test = conv2DS(X_test, y_test)

    for i in range(1, NHiddenUnits+1):
        print i
        # A neural network without a hidden layer will simulate logistic regression (albeit very slowly)
        fnn = buildNetwork(  DS_train.indim, i, DS_train.outdim, outclass=SoftmaxLayer,bias=True )
        trainer = BackpropTrainer( fnn, dataset=DS_train, momentum=0.1, verbose=False, weightdecay=0.01)
        # Train for 100 iterations.
        for tmp in range(20): trainer.trainEpochs(1)
        ote_test = fnn.activateOnDataset(DS_test)

        ErrorRate_test = (np.argmax(ote_test, 1) != y_test.T).mean(dtype=float)
        print('Error rate training (ensemble): {0}%'.format(100 * ErrorRate_test))

        ote_train = fnn.activateOnDataset(DS_train)

        ErrorRate_train = (np.argmax(ote_train, 1) != y_train.T).mean(dtype=float)
        print('Error rate test (ensemble): {0}%'.format(100 * ErrorRate_train))

        Error_train[i - 1][k], Error_test[i - 1][k] = ErrorRate_train, ErrorRate_test
        #figure(1)

        #dbplotf(X_test, y_test, neval, 'auto')
        #show()
    k += 1

figure()
plot(sum(Error_train*100, 1) / K)
plot(sum(Error_test*100, 1) / K)
xlabel('Number of hidden nodes')
ylabel('Classification error rate (%)')
legend(['Error_train', 'Error_test'])
savefig('img/Error_rate_KNN4_legend_test.pdf')
show()