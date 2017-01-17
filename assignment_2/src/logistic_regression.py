# exercise 5.2.6
from pylab import *
import sklearn.linear_model as lm
from import_data import *
from sklearn import cross_validation

def logistic_regression(X, y, class_index, class_labels, K,test_data=None):
    N, M = X.shape
    CV = cross_validation.KFold(N, K, shuffle=True)
    k = 0
    y_est_test = np.empty(K)
    y_est_train = np.empty(K)
    y = np.matrix(y).T
    X = np.matrix(X)
    for train_index, test_index in CV:
        print('Crossvalidation fold: {0}/{1}'.format(k + 1, K))
        # extract training and test set for current CV fold
        X_train = X[train_index, :]
        y_train = y[train_index, :]
        X_test =  X[test_index, :]
        y_test = y[test_index, :]

        model = lm.logistic.LogisticRegression()

        model = model.fit(X_train, y_train.A.ravel())

        y_est_test1 = model.predict_proba(X_test)[:, class_index]
        y_est_train1 = model.predict_proba(X_train)[:, class_index]

        y_est_test_error =  (y_est_test1 != y_test).sum().astype(float) / len(y_test)
        y_est_train_error = (y_est_train1 != y_test).sum().astype(float) / len(y_test)


        print y_est_test_error, y_est_train_error

    k += 1

    f = figure(); f.hold(True)
   #for label in class_labels:
   #     class_ids = nonzero(y == class_labels.index(label))[0].tolist()
   #     plot(class_ids, y_est_white_prob[class_ids], 'o')

    #class0_ids = nonzero(y == class_index)[0].tolist()
    #plot(class0_ids, y_est_white_prob[class0_ids], '.y')
    #class1_ids = nonzero(y != class_index)[0].tolist()
    #plot(class1_ids, y_est_white_prob[class1_ids], '.r')
    xlabel('Data object ' + class_labels[class_index]);
    ylabel('Predicted prob. of class ' + class_labels[class_index]);
    #legend([prediction_name, 'Not ' + prediction_name])
    legend(class_labels, loc='best')
    ylim(-0.25, 1.25)
    savefig('img/logistic_regression_of_' + class_labels[class_index] + '.pdf')
    show()



K = 10
dataObj = Data()
dataObj.normalize_data()
dataObj.X = np.hstack((dataObj.X, np.atleast_2d(dataObj.y_cont).T))

dataObj.X = dataObj.X[dataObj.X[:,-1].argsort()]
dataObj.y_cont = dataObj.X[:,-1]
dataObj.X = np.delete(dataObj.X, [-1], axis=1)

dataObj.remove_one_out_of_k(dataObj.cont_classes)

class_index = dataObj.cont_classes.index('Africa')
#print dataObj.X.sort()

#print dataObj.cont_classes[3]

logistic_regression(dataObj.X, dataObj.y_cont, class_index, dataObj.cont_classes, K)

dataObj = Data()
dataObj.normalize_data()
dataObj.X = np.hstack((dataObj.X, np.atleast_2d(dataObj.y_gov).T))

dataObj.X = dataObj.X[dataObj.X[:,-1].argsort()]
dataObj.y_gov = dataObj.X[:,-1]
dataObj.X = np.delete(dataObj.X, [-1], axis=1)

dataObj.remove_one_out_of_k(dataObj.gov_classes)
class_index = dataObj.gov_classes.index('Republic')
#logistic_regression(dataObj.X, dataObj.y_gov, class_index, dataObj.gov_classes)