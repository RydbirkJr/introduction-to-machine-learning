# exercise 2.2.3
# (requires data structures from ex. 2.2.1)
from import_data import *

from pylab import *
import scipy.linalg as linalg

# Subtract mean value from data

data = Data()
data.remove_one_out_of_k(data.cont_classes)
data.remove_one_out_of_k(data.gov_classes)

# Adjust to mean by subtracting. Moves the center of the data towards (0,0)
Y = data.X - np.ones((data.N, 1)) * data.X.mean(0)

# Standarize data by dividing by standard deviation. Notice the
# paranthesis around the (np.ones.. std(0)) - very important for result. lol
Y = Y / (np.ones((data.N, 1)) * data.X.std(0))

# PCA by computing SVD of Y
U, S, V = linalg.svd(Y, full_matrices=False)

# Compute variance explained by principal components
# Thus, rho is a 8-dimensional vector containing the percentage of the
# variance for each principal component.
rho = (S * S) / (S * S).sum()


print ''
print 'Accumulative sum of percentages:'
perc = 0
index = 1
for r in rho:
    perc += r
    print index, perc
    index += 1

for (index, att) in zip(range(len(data.attribute_names)), data.attribute_names):
    print index, att

print ''
print V[0]
print ''
print V[1]

# Plot variance explained
figure()
plot(range(1, len(rho) + 1), rho, 'o-')
title('Variance explained by principal components')
xlabel('Principal Component')
ylabel('Variance Explained')
savefig('img/pca_variance_explained.pdf')
show()
