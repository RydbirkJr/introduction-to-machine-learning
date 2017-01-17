from import_data import *
import math
from pylab import *
from sklearn import cross_validation, tree

tree_criterion = ['gini', 'entropy']


def decisionTree(criterion, X, y, file_name, attribute_names, K, tc):

    N, M = X.shape
    CV = cross_validation.KFold(N, K, shuffle=True)
    # Initialize variable
    Error_train = np.empty((len(tc), K))
    Error_test = np.empty((len(tc), K))

    Error_train2 = np.empty((len(tc), K))
    Error_test2 = np.empty((len(tc), K))
    best_dtc = None
    k = 0
    best_error_test = float("inf")
    for train_index, test_index in CV:
        print('Computing CV fold: {0}/{1}..'.format(k + 1, K))
        # extract training and test set for current CV fold
        X_train, y_train = X[train_index, :].A, y[train_index, :].A
        X_test, y_test = X[test_index, :].A, y[test_index, :].A

        for i, t in enumerate(tc):
            # Fit decision tree classifier, Gini split criterion, different pruning levels
            dtc = tree.DecisionTreeClassifier(criterion=criterion, max_depth=t)
            dtc = dtc.fit(X_train, y_train.ravel())
            y_est_test = dtc.predict(X_test)
            y_est_train = dtc.predict(X_train)
            # Evaluate misclassification rate over train/test data (in this CV fold)
            misclass_rate_test = sum(np.abs(np.mat(y_est_test).T - y_test)) / float(len(y_est_test))
            misclass_rate_train = sum(np.abs(np.mat(y_est_train).T - y_train)) / float(len(y_est_train))
            Error_test[i, k], Error_train[i, k] = misclass_rate_test, misclass_rate_train

            y_est_test2 = np.mat(dtc.predict(X_test)).T
            Error_test2[i, k] = 100 * (y_est_test2 != y_test).sum().astype(float) / len(y_test)
            y_est_train2 = np.mat(dtc.predict(X_train)).T
            Error_train2[i, k] = 100 * (y_est_train2 != y_train).sum().astype(float) / len(y_train)

        k += 1

    f = figure();
    f.hold(True)
    title(file_name)
    boxplot(Error_test.T)
    xlabel('Model complexity (max tree depth)')
    ylabel('Test error across CV folds, K={0})'.format(K))
    savefig('img/test_error_boxplot_'+file_name+'.pdf')

    f = figure();
    f.hold(True)
    title(file_name)
    plot(tc, Error_train.mean(1))
    plot(tc, Error_test.mean(1))
    xlabel('Model complexity (max tree depth)')
    ylabel('Error (misclassification rate, CV K={0})'.format(K))
    legend(['Error_train', 'Error_test'])
    savefig('img/misclassfication_rate_graph_' + file_name +'.pdf')

    f = figure();
    f.hold(True)
    title(file_name)
    plot(tc, Error_train2.mean(1))
    plot(tc, Error_test2.mean(1))
    xlabel('Model complexity (max tree depth)')
    ylabel('Error (misclassification rate, CV K={0}) (%)'.format(K))
    legend(['Error_train', 'Error_test'])
    savefig('img/misclassfication_rate_graph2_' + file_name + '.pdf')
    show()
    # Export tree graph for visualization purposes:
    # (note: you can use i.e. Graphviz application to visualize the file)
    #tree.export_graphviz(best_dtc, out_file=file_name + '.gvz', feature_names=attribute_names)

# Tree complexity parameter - constraint on maximum depth
tc = np.arange(2, 21, 1)
dataObj = Data()
dataObj.remove_one_out_of_k(dataObj.gov_classes)
dataObj.X = np.matrix(dataObj.X)
dataObj.y_gov = np.matrix(dataObj.y_gov).T

#decisionTree(tree_criterion[0], dataObj.X, dataObj.y_gov, 'Government_Gini', dataObj.attribute_names, 40, tc)
decisionTree(tree_criterion[1], dataObj.X, dataObj.y_gov, 'Government_Entropy', dataObj.attribute_names, 40, tc)

dataObj = Data()
dataObj.normalize_data()
dataObj.remove_one_out_of_k(dataObj.cont_classes)
dataObj.X = np.matrix(dataObj.X)
dataObj.y_cont = np.matrix(dataObj.y_cont).T

#Classification of continent
#decisionTree(tree_criterion[0], dataObj.X, dataObj.y_cont, 'Continent_Gini', dataObj.attribute_names, 40, tc)
#decisionTree(tree_criterion[1], dataObj.X, dataObj.y_cont, 'Continent_Entropy', dataObj.attribute_names, 40, tc)