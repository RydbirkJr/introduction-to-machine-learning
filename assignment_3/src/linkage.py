# exercise 10.2.1
from pylab import *
from import_data import *
from scipy.io import loadmat
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

data = Data()

data.normalize_data()
classNames = data.gov_classes

data.remove_one_out_of_k(classNames)
y = np.matrix(data.y_gov).T
attributeNames = data.attribute_names
X = data.get_pca()[:,:2] #= np.matrix(data.X)
#N, M = data.N, data.M

C = len(classNames)


# Perform hierarchical/agglomerative clustering on data matrix


def plot_function(Method):
    Metric = 'euclidean'
    Z = linkage(data.X, method=Method, metric=Metric)

    # Compute and display clusters by thresholding the dendrogram
    Maxclust = 3
    cls = fcluster(Z, criterion='maxclust', t=Maxclust)

    figure(1)
    clusterplot(X, cls.reshape(cls.shape[0],1), y=y, classes=classNames)
    title(Method.title(), fontweight='bold')
    ylabel('PC2')
    xlabel('PC1')
    savefig('img/cluster_' + method + '.pdf')

    # Display dendrogram
    max_display_levels=10
    figure(2)
    dendrogram(Z, truncate_mode='level', p=max_display_levels)
    title(Method.title(), fontweight='bold')
    ylabel('Distance')
    xlabel('Country')
    savefig('img/hiearchy_' + method + '.pdf')
    show()

#  , 'average', 'weighted', 'centroid', 'median' produces the same as complete
methods = ['single', 'complete', 'ward']

for method in methods:
    plot_function(method)
