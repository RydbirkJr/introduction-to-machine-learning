from import_data import *
import math

data = Data()


#print 'Government type: ', data.gov_classes
#print 'Continent: ', data.cont_classes
#print 'Attribute: ', data.attribute_names


#data.normalize_data()
#data.remove_and_get_y('gdp_per_cap')

#print data.attribute_names


index = data.attribute_names.index('population')

print data.X[0]
print ''
X = data.X[:, index]

#X = np.sqrt(X)
X = np.log(X)

data.X[:, index] = X

print data.X[0]
