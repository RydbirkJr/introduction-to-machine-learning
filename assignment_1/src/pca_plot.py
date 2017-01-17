# exercise 2.2.4
# (requires data structures from ex. 2.2.1 and 2.2.3)
from import_data import *
from pylab import *

data = Data()
cont_index = data.attribute_names.index('continent')
gov_index = data.attribute_names.index('gov_type')


def plot_data_on_pca(pca1, pca2, classes, class_index, type):
    Y = data.X - np.ones((data.N, 1)) * data.X.mean(0)
    Y = Y / (np.ones((data.N, 1)) * data.X.std(0))

    # PCA by computing SVD of Y
    U, S, V = linalg.svd(Y, full_matrices=False)
    V = mat(V).T

    # Project the centered data onto principal component space
    Z = Y * V

    f = figure()
    f.hold()
    title('PCA')
    Z = array(Z)
    for c in range(len(classes)):
        class_mask = data.X[:, class_index].ravel() == c
        # select indices belonging to class c:
        plot(Z[class_mask, pca1], Z[class_mask, pca2], 'o')

    legend(classes, loc='upper left')
    xlabel('PC{0}'.format(pca1 + 1))
    ylabel('PC{0}'.format(pca2 + 1))

    savefig('img/{2}_pca_{0}_pca_{1}.pdf'.format(pca1+1, pca2+1, type))

    # Output result to screen
    #show()


def plot_data_on_pca_both_classes(pca1, pca2):
    plot_data_on_pca(pca1, pca2, data.cont_classes, cont_index, 'cont')
    # plot_data_on_pca(pca1, pca2, data.gov_classes, gov_index, 'gov')


# plot_data_on_pca_both_classes(0,1)
# plot_data_on_pca_both_classes(0,2)
# plot_data_on_pca_both_classes(1,2)
# plot_data_on_pca_both_classes(0,3)
plot_data_on_pca_both_classes(1,2)
# plot_data_on_pca_both_classes(2,3)
